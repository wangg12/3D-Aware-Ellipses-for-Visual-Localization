import argparse
import glob
import logging
import json
import os
import random
import time

import cv2
import numpy as np

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from config import _cfg
from dataset_loader import Dataset_loader
from utils import create_if_needed



def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="<Required> Input dataset file (.json).")
    parser.add_argument("detector_checkpoint", help="<Required> Detector checkpoint.")
    parser.add_argument("output", help="<Required> Folder where to output the images with detections.")
    parser.add_argument("--save_detections_file", default=None,
                        help="<Optional> File where to write the detections (.json).")
    parser.add_argument("--visualize",  action="store_true", default=False,
                        help="<Optional> Visualize the object detected in each image (default is None).")
    parser.add_argument("--skip_frames", default=0, type=int,
                        help="<Optional> Skip frames to process (default is 0, no skipping).")
    args = parser.parse_args(args)


    dataset = args.dataset
    detector_checkpoint = args.detector_checkpoint
    output_folder = args.output
    output_file = args.save_detections_file
    visualize = args.visualize
    skip_frames = args.skip_frames

    create_if_needed(output_folder)

    loader = Dataset_loader(dataset)


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(_cfg.DETECTOR_ARCHITECTURE))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = _cfg.DETECTOR_NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = _cfg.DETECTOR_SCORE_THRESH_TEST
    cfg.MODEL.WEIGHTS = detector_checkpoint
    predictor = DefaultPredictor(cfg)


    pred_data = []
    for idx in range(0, len(loader), 1+skip_frames):
        f = loader.get_rgb_filename(idx)
        name = os.path.splitext(os.path.basename(f))[0]
        im = cv2.imread(f)

        predictions = predictor(im)
        v = Visualizer(im[:, :, ::-1], scale=1.0)
        out = v.draw_instance_predictions(predictions["instances"].to("cpu"))

        if visualize:
            cv2.imshow("fen", out.get_image()[:, :, ::-1])
            cv2.waitKey(-1)

        cv2.imwrite(os.path.join(output_folder, name + ".png"), out.get_image()[:, :, ::-1])

        instances = predictions["instances"]
        classes = instances.get("pred_classes").cpu().numpy().astype(int)
        scores = instances.get("scores").cpu().numpy()
        boxes = instances.get("pred_boxes").tensor.cpu().numpy()
        print("(%d / %d) => " % (idx, len(loader)), len(boxes), "detections")

        detections = []
        index = 0
        for c, s, b in zip(classes, scores, boxes):
            if s >= cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                det = {"category_id": int(c), "score": float(s), "bbox":b.tolist()}
                detections.append(det)
        pred_data.append({"file_name": f, "detections": detections})

        print("\rProgress:  ", idx, "/", len(dataset), end="", flush=True)

    if output_file is not None:
        with open(output_file, "w") as fout:
            json.dump(pred_data, fout)
        print("\nPredictions saved in", output_file)



if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)