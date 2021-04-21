import argparse
import json
import logging
import os
import random

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer

from config.config import _cfg


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="<Required> Input training dataset file (.json).")
    parser.add_argument("output", help="<Required> Output checkpoint file (.pth).")
    parser.add_argument("--nb_epochs", help="Number of training epochs.",
                        default=_cfg.DETECTOR_NB_EPOCHS, type=int)
    args = parser.parse_args(args)


    training_data = args.dataset
    output_file = args.output
    output_folder = os.path.dirname(output_file)
    if len(output_folder) == 0:
        output_folder = "./"
    nb_epochs = args.nb_epochs

    def my_dataset():
        with open(training_data, "r") as fin:
            data = json.load(fin)
        return data

    DatasetCatalog.register("custom_dataset", my_dataset)
    metadata = MetadataCatalog.get("custom_dataset")
    dataset_dicts = my_dataset()



    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(_cfg.DETECTOR_ARCHITECTURE))
    cfg.DATASETS.TRAIN = ("custom_dataset",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = _cfg.DETECTOR_DATALOADER_NUM_WORKERS
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(_cfg.DETECTOR_ARCHITECTURE)
    cfg.SOLVER.IMS_PER_BATCH = _cfg.DETECTOR_IMS_PER_BATCH
    cfg.SOLVER.BASE_LR = _cfg.DETECTOR_BASE_LR
    cfg.SOLVER.MAX_ITER = nb_epochs
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = _cfg.DETECTOR_BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = _cfg.DETECTOR_NUM_CLASSES
    cfg.OUTPUT_DIR = output_folder


    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    os.rename(os.path.join(output_folder, "model_final.pth"), output_file )


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)