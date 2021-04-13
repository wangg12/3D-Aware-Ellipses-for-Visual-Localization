#!/usr/bin/env python3
"""
@author: mzins
"""

import argparse
import json
import logging
import os

import cv2
import numpy as np

from ellcv.types import Ellipsoid, Ellipse
from ellcv.utils import bbox_from_ellipse
from ellcv.visu import draw_bbox, draw_ellipse

from dataset_loader import Dataset_loader
from scene_loader import Scene_loader
from utils import create_if_needed

"""
    Generate object annotations.
"""


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", help="<Required> Input Scene file (.json)")
    parser.add_argument("dataset", help="<Required> Input Dataset file (.json)")
    parser.add_argument("output", help="<Required> Output annotated file (.json)")
    parser.add_argument("--output_images", default=None,
                        help="<Optional> Output folder where to write the annotated images (default is None).")
    parser.add_argument("--visualize",  action="store_true", default=False,
                        help="<Optional> Visualize the annotated images (default is None)")
    args = parser.parse_args(args)

    input_dataset_file = args.dataset
    input_scene_file = args.scene
    output_file = args.output
    output_folder = args.output_images
    visualize = args.visualize

    if output_folder is not None:
        create_if_needed(output_folder)

    # Load scene
    scene = Scene_loader(input_scene_file)

    # Load dataset
    loader = Dataset_loader(input_dataset_file)


    images_annotations = []
    for idx in range(len(loader)):
        width, height = loader.get_image_size(idx)
        Rt = loader.get_Rt(idx)
        K = loader.get_K(idx)
        P = K @ Rt

        img_data = {
            "file_name": loader.get_rgb_filename(idx),
            "width": width,
            "height": height,
            "K": K.tolist(),
            "R": Rt[:3, :3].tolist(),
            "t": Rt[:3, -1].tolist()
            }
        
        annotations = []
        for obj in scene:
            ell = obj["ellipsoid"].project(P)

            bbox = bbox_from_ellipse(ell)
            bbox = np.round(bbox).astype(int).tolist()

            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] >= width or bbox[3] >= height:
                continue

            annot = {
                "bbox": bbox,
                "bbox_mode": 0,
                "object_id": obj["object_id"],
                "category_id": obj["category_id"],
                "ellipse": ell.as_dual().tolist()
                }
            annotations.append(annot)
        img_data["annotations"] = annotations
        images_annotations.append(img_data)

        if visualize or output_folder is not None:
            f = loader.get_rgb_filename(idx)
            img = cv2.imread(f)
            for obj in annotations:
                bbox = obj["bbox"]
                ell = Ellipse.from_dual(np.asarray(obj["ellipse"]))
                draw_bbox(img, bbox, color=(255, 255, 255))
                draw_ellipse(img, ell)
            if visualize:
                cv2.imshow("viz", img)
                cv2.waitKey()
            if output_folder is not None:
                name = ("frame_%04d_" % idx) + os.path.splitext(os.path.basename(f))[0]
                cv2.imwrite(os.path.join(output_folder, name + ".png"), img)
        
    with open(output_file, "w") as fout:
        json.dump(images_annotations, fout)

    print("Object annotated for dataset (%s) saved in: %s" % (input_dataset_file, output_file))


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)