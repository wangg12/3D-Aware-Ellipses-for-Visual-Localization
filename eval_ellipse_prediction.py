import argparse
import json
import logging
import os
import time

import torch
from torchvision import transforms
import pytorch_lightning as pl

from config.config import _cfg
from dataset.dataset import EllipsesDataset
from dataset.scene_loader import Scene_loader
from ellipse_prediction.model import EllipsePredictor



def run_evaluation(validation_dataset_file, object_name, checkpoint_path, output_folder=None):
    """
        Run the evaluation for a given object id
    """
    print("=============================================")
    print("  Start evaluating for object: %d (category %d)" % (object_name[1], object_name[0]))
    print("=============================================")
    category_id, object_id = object_name
    name = "obj_%02d_%02d" % (category_id, object_id)
    dataset_valid = EllipsesDataset(validation_dataset_file, object_id, crop_size=256,
                                    transforms=transforms.Compose([
                                        transforms.ColorJitter(*_cfg.VALID_COLOR_JITTER),
                                        transforms.ToTensor()]),
                                    random_shift=_cfg.VALID_RANDOM_SHIFT,
                                    random_rotation=_cfg.VALID_RANDOM_ROTATION,
                                    random_perspective=_cfg.VALID_RANDOM_PERSPECTIVE,
                                    random_blur=_cfg.VALID_RANDOM_BLUR,
                                    probability_do_augm=_cfg.VALID_PROBA_AUGM)

    validation_loader = torch.utils.data.DataLoader(dataset_valid, 
                                                    batch_size=_cfg.VALID_BATCH_SIZE,
                                                    shuffle=_cfg.VALID_BATCH_SHUFFLE,
                                                    num_workers=_cfg.VALID_LOADER_NUM_WORKERS)
    if output_folder is not None:
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

    # Model
    model = EllipsePredictor(crop_size=_cfg.CROP_SIZE,
                             output_val_images=output_folder)
    
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    if os.path.splitext(checkpoint_path)[1] == ".pth":
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint["state_dict"])


    # Testing
    ta = time.time()
    trainer = pl.Trainer(gpus=1)

    trainer.test(model, validation_loader)

    print("Finished in %.2fs" % (time.time()-ta))


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", help="<Required> Input scene file containing the objects (.json)")
    parser.add_argument("dataset", help="<Required> Input evaluation dataset file (.json)")
    parser.add_argument("ckpts", help="<Required> Folder containing the checkpoints")
    parser.add_argument("--mode", help="<Optional> Choose between 'best'(default) or 'last' checkpoint", default="best")
    parser.add_argument("--output_images", help="<Optional> Output folder where to save "
                                                "the images with the predicted ellipses.", default=None)
    parser.add_argument("--object_id", help="<Optional> Specify a single object id for evaluating.",
                        default=None, type=int)
    args = parser.parse_args(args)


    eval_dataset_file = args.dataset
    scene_file = args.scene
    checkpoints_folder = args.ckpts
    mode = args.mode
    output_images = args.output_images
    only_object_id = args.object_id

    scene = Scene_loader(scene_file)

    if only_object_id is not None:
        scene = scene[only_object_id:only_object_id+1]

    for obj in scene:
        cat_id = obj["category_id"]
        obj_id = obj["object_id"]
        ckpt_file = os.path.join(checkpoints_folder, "obj_%02d_%02d" % (cat_id, obj_id),
                                 "ckpt-" + mode + ".ckpt")
        out = None
        if output_images is not None:
            out = os.path.join(output_images, "obj_%02d_%02d" % (cat_id, obj_id))
        run_evaluation(eval_dataset_file, (cat_id, obj_id), ckpt_file, out)



if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)