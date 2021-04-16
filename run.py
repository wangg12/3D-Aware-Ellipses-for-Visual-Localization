import argparse
import glob
import logging
import json
import os
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from ellcv.types import Ellipse, Ellipsoid
from ellcv.utils import pose_error
from ellcv.visu import draw_ellipse, draw_bbox

from config import _cfg
from dataset_loader import Dataset_loader
from model import EllipsePredictor
from pose_computation import compute_pose
from scene_loader import Scene_loader
from utils import create_if_needed, force_square_bbox


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", help="<Required> Input Scene file containing the objects (.json).")
    parser.add_argument("dataset", help="<Required> Dataset to process (.json).")
    parser.add_argument("detector_checkpoint", help="<Required> Checkpoint for the detection network.")
    parser.add_argument("ellipses_checkpoints", help="<Required> Folder containing the checkpoints for ellipse predictions")

    parser.add_argument("--mode", default="best",
                        help="Choose between 'best'(default) or 'last' checkpoint for ellipse prediction")
    parser.add_argument("--output_predictions", default=None,
                        help="<Optional> Output file containing the detected objects and predicted ellipses")
    parser.add_argument("--output_images", default=None,
                        help="<Optional> Output folder where to write the images with predictions and reporjected objects (default is None).")
    parser.add_argument("--output_errors", default=None,
                        help="<Optional> Output folder where to write the position and orientation errors (default is None).")
    parser.add_argument("--visualize",  action="store_true", default=False,
                        help="<Optional> Visualize each predictions and estimated poses online (default is None)")
    parser.add_argument("--only_prediction",  action="store_true", default=False,
                        help="<Optional> Do only object detection and ellipse prediction"
                            "without computing the camera pose (defaut is False).")
    parser.add_argument("--skip_frames", default=0, type=int,
                        help="<Optional> Skip frames to process (default is 0, no skipping).")
    parser.add_argument('--min_obj_for_P3P', choices=[3, 4], type=int, default=4,
                        help="<Optional> Minimum number of required detected objects to use P3P."
                            "When less (but >= 2) P2E can be used. (default is 4)")
    args = parser.parse_args(args)



    scene_file = args.scene
    dataset_file = args.dataset
    detector_checkpoint = args.detector_checkpoint
    checkpoints_folder_ellipses = args.ellipses_checkpoints
    mode = args.mode
    output_file = args.output_predictions
    output_folder = args.output_images
    output_errors = args.output_errors
    visualize = args.visualize
    skip_frames = args.skip_frames
    do_pose_computation = not args.only_prediction
    min_obj_for_P3P = args.min_obj_for_P3P


    if output_folder is not None:
        create_if_needed(output_folder)
    if do_pose_computation and output_errors:
        create_if_needed(output_errors)

    # Load the scene
    scene = Scene_loader(scene_file)

    # Load the dataset
    loader = Dataset_loader(dataset_file)
    K = loader.get_K(0)


    # Configure Object detector
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(_cfg.DETECTOR_ARCHITECTURE))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = _cfg.DETECTOR_NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = _cfg.DETECTOR_SCORE_THRESH_TEST
    cfg.MODEL.WEIGHTS = detector_checkpoint
    predictor = DefaultPredictor(cfg)



    # Configure Ellipse predictor
    device = torch.device("cuda")
    models = {}
    for obj in scene:
        cat_id = obj["category_id"]
        obj_id = obj["object_id"]
        ckpt_file = os.path.join(checkpoints_folder_ellipses,
                                "obj_%02d_%02d" % (cat_id, obj_id), "ckpt-" + mode + ".ckpt")
        ckpt_file = glob.glob(os.path.join(checkpoints_folder_ellipses, "obj_%02d_%02d" % (cat_id, obj_id), "*.pth"))[0]
        model = EllipsePredictor().to(device)
        model.load(ckpt_file)
        if cat_id in models.keys():
            models[cat_id].append(model)
        else:
            models[cat_id] = [model]


    # Processing loop
    output_data = []
    position_errors = [-1] * len(loader)
    orientation_errors = [-1] * len(loader)
    for idx in range(0, len(loader), 1+skip_frames):
        print("({:4} / {}) => ".format(idx, len(loader)), end=" ")
        f = loader.get_rgb_filename(idx)

        # Read image twice (cv2 and PIL) to ensure each network has the correct input format
        img = cv2.imread(f)
        img_pil = Image.open(f)


        # Run object detection
        predictions = predictor(img)

        instances = predictions["instances"]
        classes = instances.get("pred_classes").cpu().numpy().astype(int)
        scores = instances.get("scores").cpu().numpy()
        boxes = instances.get("pred_boxes").tensor.cpu().numpy()
        detections_txt = []
        detections = []
        for c, s, b in zip(classes, scores, boxes):
            if s > cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                det = {"category_id": int(c), "score": float(s), "bbox":b.tolist()}
                x1, y1, x2, y2 = force_square_bbox(b, margin=2)
                crop = img_pil.crop((x1, y1, x2, y2))
                if c not in models.keys():
                    continue
                # Run ellipse prediction
                pred_ellipses_txt = []
                pred_ellipses = []
                for model in models[c]:
                    axes, angle, center = model.predict(crop, device)
                    center += np.array([x1, y1])
                    ellipse = Ellipse.compose(axes, angle, center)
                    pred_ellipses_txt.append(ellipse.to_dict())
                    pred_ellipses.append(ellipse)
                det["ellipses"] = pred_ellipses_txt
                detections_txt.append(det)
                detections.append({"category_id": int(c),
                                "bbox": b.tolist(), 
                                "ellipses" : pred_ellipses})

        out_data = {"file_name": f, 
                    "bbox": b.tolist(),
                    "detections": detections_txt,
                    "orientation": [],
                    "position": []
                    }


        if do_pose_computation:
            # Compute camera pose
            pose, used_pairs, inliers = compute_pose(detections, scene, K, min_obj_for_P3P=min_obj_for_P3P)
            if pose is None:
                output_data.append(out_data)
                continue
            o = pose[:3, :3]
            p = pose[:, 3]
            P = K @ np.hstack((o.T, (-o.T @ p).reshape((-3, 1))))

            # Evaluate pose error
            Rt_gt = loader.get_Rt(idx)
            orientation_gt = Rt_gt[:3, :3].T
            position_gt = -Rt_gt[:3, :3].T @ Rt_gt[:, 3]
            rot_error, pos_error = pose_error((o, p), (orientation_gt, position_gt))
            print("Estimated pose error: %.3fm %.2f°" % (pos_error, np.rad2deg(rot_error)))
            
            out_data["orientation"] = o.tolist()
            out_data["position"] = p.tolist()
            position_errors[idx] = pos_error
            orientation_errors[idx] = np.rad2deg(rot_error)

        output_data.append(out_data)
        


        if visualize or output_folder is not None:
            if do_pose_computation:
                # Display objects detections, ellipses predictions and reprojected 
                # ellipsoids using the estimated pose
                used_ellipses = set([p[0] for p in used_pairs])
                used_ellipsoids = set([p[1] for p in used_pairs])
                inlier_ellipses = set([p[0] for p in inliers])
                inlier_ellipsoids = set([p[1] for p in inliers])
                for det_i, det in enumerate(detections):
                    bb = np.round(det["bbox"]).astype(int)
                    draw_bbox(img, bb, color=(255, 255, 255))
                    for pred_i, ell in enumerate(det["ellipses"]):
                        pred = (det_i, pred_i)
                        if pred in used_ellipses:
                            col = (0, 255, 0)
                        elif pred in inlier_ellipses:
                            col = (255, 0, 0)
                        else:
                            col = (0, 0, 255)
                        draw_ellipse(img, ell, col, 2)
                    cv2.putText(img, "det %d" % det_i, (bb[0], bb[1] + 20), cv2.FONT_HERSHEY_SIMPLEX , 0.5, col, 2, cv2.LINE_AA)

                for obj_i, obj in enumerate(scene):
                    proj_ell = obj["ellipsoid"].project(P)
                    if obj_i in used_ellipsoids:
                        col = (0, 255, 0)
                    elif obj_i in inlier_ellipsoids:
                        col = (255, 0, 0)
                    else:
                        col = (0, 0, 255)
                    draw_ellipse(img, proj_ell, col, 4)
                    cent = proj_ell.center
                    cv2.putText(img, str(obj["category_id"]), (int(cent[0]), int(cent[1])), cv2.FONT_HERSHEY_SIMPLEX , 1, col, 2, cv2.LINE_AA)
                    cv2.putText(img, str(obj["object_id"]), (int(cent[0]), int(cent[1])+20), cv2.FONT_HERSHEY_SIMPLEX , 1, col, 2, cv2.LINE_AA)
                cv2.putText(img, "%.3fm" % pos_error, (25, 30), cv2.FONT_HERSHEY_SIMPLEX , 1, (50, 255, 50), 2, cv2.LINE_AA)
                cv2.putText(img, "%.2fdeg" % np.rad2deg(rot_error), (475, 30), cv2.FONT_HERSHEY_SIMPLEX , 1, (50, 255, 50), 2, cv2.LINE_AA)
            else:
                # Display only detections and predicted ellipses
                v = Visualizer(img, scale=1.0)
                out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
                img = out.get_image().copy()
                for det in detections:
                    for ell in det["ellipses"]:
                        draw_ellipse(img, ell)

            if visualize:
                cv2.imshow("fen", img)
                cv2.waitKey(-1)
            if output_folder is not None:
                name = ("frame_%04d_" % idx) + os.path.splitext(os.path.basename(f))[0]
                cv2.imwrite(os.path.join(output_folder, name + ".png"), img)

    # Save the detections and predicted ellipses
    if output_file is not None:
        with open(output_file, "w") as fout:
            json.dump(output_data, fout)

    # Save errors files
    if output_errors and do_pose_computation:
        np.savetxt(os.path.join(output_errors, "rot_errors.txt"), orientation_errors)
        np.savetxt(os.path.join(output_errors, "pos_errors.txt"), position_errors)
        print("Errors files saved in", output_errors)

    if output_file is not None:
        print("\nOutput predictions and poses saved in", output_file)


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)