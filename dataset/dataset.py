import argparse
import json
import logging
import sys

import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms

from ellcv.types import Ellipse
from ellcv.visu import draw_ellipse

sys.path.append("../utils")
from utils.utils import force_square_bbox


class EllipsesDataset(torch.utils.data.Dataset):
    
    def __init__(self, annotation_file, object_id, transforms, 
                 random_shift=None, random_rotation=None,
                 random_perspective=None, random_blur=None,
                 probability_do_augm=0.5, crop_size=256, crop_margin=2):
        self.annotation_file = annotation_file
        self.object_id = object_id
        self.load_annotations()
        self.crop_margin = crop_margin
        self.crop_size = crop_size
        
        self.transforms = transforms
        self.random_blur = random_blur
        self.random_shift = random_shift
        self.random_rotation = random_rotation
        self.random_perspective = random_perspective
        self.proba_augm = probability_do_augm
        
    def load_annotations(self):        
            with open(self.annotation_file, "r") as fin:
                self.annotations = json.load(fin)
            self.images_filename = []
            self.bboxes = []
            self.ellipses = []
            
            for img_annot in self.annotations:
                for annot in img_annot["annotations"]:
                    if annot["object_id"] == self.object_id:
                        self.images_filename.append(img_annot["file_name"])
                        self.bboxes.append(annot["bbox"])
                        self.ellipses.append(Ellipse.from_dict(annot["ellipse"]))
                        continue
            
    
    def __len__(self):
        return len(self.images_filename)
    
    def __getitem__(self, idx):
        img = Image.open(self.images_filename[idx])
        ell = self.ellipses[idx]
        bbox = self.bboxes[idx] 
        bbox = np.array(bbox).reshape((2, 2))
        

        # Use a temporary crop around the bbox to avoid rescaling the whole image
        # This way the bbox can be translated/deformed without needing to fill
        # with other pixels
        temp_bbox = force_square_bbox(bbox, 100)
        first_corner = np.array(temp_bbox[:2])
        img = img.crop(temp_bbox)
        bbox -= first_corner
        ell = ell.translate(-first_corner)

        # Rescale the square bbox to crop_size
        sq_bbox = force_square_bbox(bbox, self.crop_margin)
        f = self.crop_size / (sq_bbox[2] - sq_bbox[0])
        ell_resized = ell.full_scale(f)
        sq_bbox_resized = np.array(sq_bbox) * f
        img_resized = img.resize(map(lambda x: int(round(f*x)), img.size),
                                  resample=Image.BICUBIC)

        # Crop box
        cb = np.array([[sq_bbox_resized[0], sq_bbox_resized[1]],
                       [sq_bbox_resized[0], sq_bbox_resized[3]],
                       [sq_bbox_resized[2], sq_bbox_resized[1]],
                       [sq_bbox_resized[2], sq_bbox_resized[3]]],
                       dtype=float)
        
        # Crop box deformation for data augmentation
        do_shift = np.random.rand(1).item() < self.proba_augm and self.random_shift is not None
        do_rotation = np.random.rand(1).item() < self.proba_augm and self.random_rotation is not None
        do_perspective = np.random.rand(1) < self.proba_augm and self.random_perspective is not None
        do_blur = np.random.rand(1) < self.proba_augm and self.random_blur is not None

            
        # Perspective deformation
        if do_perspective:
            # corners_shift = np.random.randn(4, 2) * self.random_perspective
            corners_shift = np.random.uniform(self.random_perspective[0],
                                              self.random_perspective[1]+1,
                                              size=(4, 2))
            cb += corners_shift
        
        # Rotation (wrt. the box center)
        if do_rotation:
            rot = np.deg2rad(np.random.uniform(self.random_rotation[0], 
                                               self.random_rotation[1]+1))
            R = np.array([[np.cos(rot), -np.sin(rot)],
                          [np.sin(rot), np.cos(rot)]])
            m = np.mean(cb, axis=0).reshape((1, 2))
            cb = (R @ (cb-m).T).T + m

        # Shift        
        if do_shift:
            shift = np.random.randint(self.random_shift[0], 
                                      self.random_shift[1]+1, size=(1, 2))
            cb += shift

    
        # Find the homography to transfrom the crop box into canonical rectangle
        # The inverse transform is computed (form corners_out to corners_in) 
        # because it is needed to transform the image the ellipse
        corners_in = cb.astype(np.float32)       
        corners_out = np.array([[0, 0],
                                [0, self.crop_size-1],
                                [self.crop_size-1, 0],
                                [self.crop_size-1, self.crop_size-1]],
                               dtype=np.float32)
        H = cv2.getPerspectiveTransform(corners_out, corners_in)
        
        # Transform the image and the ellipse
        img_transf = img_resized.transform((self.crop_size, self.crop_size),
                                           Image.PERSPECTIVE, H.flatten(),
                                           fillcolor=(0, 255, 0),
                                           resample=Image.BICUBIC)
        ell_transf = ell_resized.perspective_transform(H)
        
        
        if do_blur:
            kernel_size = np.random.randint(1, self.random_blur)
            img_transf = img_transf.filter(ImageFilter.GaussianBlur(kernel_size))


        if self.transforms:
            img_transf = self.transforms(img_transf)
        
        axes, angle, center = ell_transf.decompose()
        ell_params = np.hstack((axes / self.crop_size, center / self.crop_size, np.sin(angle))).astype(np.float32)
        
        return img_transf, ell_params
    
        

def test(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="<Required> Input dataset file (.json)")
    parser.add_argument("object_id", help="<Required> Object_id", type=int)

    parser.add_argument("--color_jitter", nargs=4, type=float,
                        metavar=("brightness", "contrast", "saturation", "hue"),
                        default=[0.0, 0.0, 0.0, 0.0],
                        help="<Optional> Color jitter to apply.")
    parser.add_argument("--shift", nargs=2, type=float,
                        metavar=("min", "max"), default=None,
                        help="<Optional> Random shift to apply.")
    parser.add_argument("--rotation", nargs=2, type=float,
                        metavar=("min", "max"), default=None,
                        help="<Optional> Random rotation to apply.")
    parser.add_argument("--perspective", nargs=2, type=float,
                        metavar=("min", "max"), default=None,
                        help="<Optional> Random perspective deformation to apply.")
    parser.add_argument("--blur", type=float, default=None,
                        help="<Optional> Random blur to apply.")
    parser.add_argument("--proba_do_augm", type=float, default=0.5,
                        help="<Optional> Probability of applying each of the augmentations.")

                        

    args = parser.parse_args(args)

    dataset = args.dataset
    object_id = args.object_id
    color_jitter = args.color_jitter
    shift = args.shift
    rotation = args.rotation
    perspective = args.perspective
    blur = args.blur
    prob_to_augm = args.proba_do_augm


    dataset = EllipsesDataset(dataset, object_id,
                              transforms=transforms.Compose([
                                  transforms.ColorJitter(*color_jitter),
                                  transforms.ToTensor()]),
                              random_shift=shift,
                              random_rotation=rotation,
                              random_perspective=perspective,
                              random_blur=blur,
                              probability_do_augm=prob_to_augm)

    for i in range(len(dataset)):
        print(i)
        img, ell_params = dataset[i]
        ell_params[:4] *= 256
        ell = Ellipse.compose(ell_params[:2], np.arcsin(ell_params[4]), ell_params[2:4])
        img_np = (img.permute((1, 2, 0)).numpy() * 255).astype(np.uint8)
        img_cv = np.array(img_np)[:, :, ::-1].copy()
        draw_ellipse(img_cv, ell, color=(0, 0, 255))
        cv2.imshow("viz", img_cv)
        cv2.waitKey()



if __name__ == '__main__':
    import sys
    try:
        test(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)