#!/usr/bin/env python3
"""
@author: Matthieu Zins
"""

import argparse
import glob
import json
import logging
import numpy as np
import os

"""
    Pre-process the 7-Scenes dataset to generate a dataset file in JSON format.
"""


# ===== Correction transform =====
# This transform goes from new basis to 7-Scenes chess basis.
# The new basis is a rotated version of the initial one, so that the axis Z is
# aligned with the vertical of the scene.
R_corr = np.array([[9.999979570965712439e-01, 0.000000000000000000e+00, 2.021336855660396583e-03],
                  [1.976414712027970994e-03, -2.096522088946129736e-01, -9.777740255756928178e-01],
                  [4.237777367092936316e-04, 9.777760230776829653e-01, -2.096517805953965752e-01]])
t_corr = np.array([2.316996306835403371e-04, -1.120792311204506370e-01, -2.403173919283707952e-02])


# Estimated intrinsics (as not provided in the dataset)
K = np.array([[532.57, 0.0, 320],
                [0.0, 531.54, 240],
                [0.0, 0.0, 1.0]])
# Images size
w = 640
h = 480


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", nargs='+', required=True,
                        help="<Required> 7-Scenes sequences folders.")
    parser.add_argument("-o", "--output", required=True,
                        help="Output dataset file (.json)")
    args = parser.parse_args(args)

    input_folders = args.input
    output_dataset_file = args.output

    # Read images and pose files
    image_files = []
    pose_files = []
    for folder in input_folders:
        image_files.extend(sorted(glob.glob(os.path.join(folder, "frame-*.color.png"))))
        pose_files.extend(sorted(glob.glob(os.path.join(folder, "frame-*.pose.txt"))))


    data = []
    for img_f, pose_f in zip(image_files, pose_files):
        Rt_inv = np.loadtxt(pose_f)
        R = Rt_inv[:3, :3].T
        t = -R@Rt_inv[:3, 3]
        t_corrected = R @ t_corr + t
        R_corrected = R @ R_corr
        d = {
            "file_name": img_f,
            "width": w,
            "height": h,
            "K": K.tolist(),
            "R": R_corrected.tolist(),
            "t": t_corrected.tolist()
        }
        data.append(d)

    with open(output_dataset_file, "w") as fout:
        json.dump(data, fout)

    print(*input_folders, sep="\n")
    print("Dataset file from the selected sequences saved in:", output_dataset_file)

if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)