import numpy as np

from ellcv.algo.cpp import solveP2E_ransac, solveP3P_ransac
from ellcv.types import Ellipsoid, Ellipse


def compute_pose(detections, scene, K, min_obj_for_P3P=4):
    """
        Compute the camera pose using objects detections and a known scene model.
        Parameters:
            - detections: list of detections:
                [
                    {
                        "category_id": ...,
                        "bbox": [xmin, ymin, xmax, ymax],
                        "ellipses": [ Ellipse, Ellipse, Ellipse, ...]
                    }, ...
                ]
            - scene: scene loader
            - K: intrinsic matrix of the camera
            - min_obj_for_P3P: minimum number of objects needed to use P3P,
                               otherwise P2E can be used.
    """

    # Find mapping between detections and objects classes
    mapping_det_to_obj = [[] for i in range(len(detections))]
    for di, d in enumerate(detections):
        for oi, o in enumerate(scene):
            if o["category_id"] == d["category_id"]:
                mapping_det_to_obj[di].append(oi)

    if len(mapping_det_to_obj) < 2:
        print("Pose computation failed: Not enough objects can be used.")
        return None, None, None


    ellipsoids_categories = [obj["category_id"] for obj in scene]
    ellipsoids_duals = [obj["ellipsoid"].as_dual() for obj in scene]

    detections_categories = [det["category_id"] for det in detections]
    detections_bboxes = [np.asarray(d["bbox"]).reshape((2, 2)) for d in detections]
    detections_ellipses_duals = [[ell.as_dual() for ell in pred["ellipses"]] for pred in detections]
    
        
    if len(detections_bboxes) >= min_obj_for_P3P:
        best_index, poses, scores, used_pairs, inliers = solveP3P_ransac(
            ellipsoids_duals, ellipsoids_categories, detections_bboxes,
            detections_categories, detections_ellipses_duals, K)
    else:
        best_index, poses, scores, used_pairs, inliers = solveP2E_ransac(
            ellipsoids_duals, ellipsoids_categories, detections_bboxes,
            detections_categories, detections_ellipses_duals, K)

    if best_index < 0:
        print("Pose computation failed: No valid pose could be obtained.")
        return None, None, None

    return poses[best_index], used_pairs[best_index], inliers[best_index]
