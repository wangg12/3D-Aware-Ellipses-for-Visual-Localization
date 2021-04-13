import numpy as np
import json

class Dataset_loader:
    """
        Interace with our own dataset file format.
    """

    def __init__(self, filename):
        with open(filename, "r") as fin:
            self.dataset = json.load(fin)
        self.nb_images = len(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def get_image_size(self, idx):
        """
            Get the image size.
        """
        if idx < 0 or idx >= self.nb_images:
            print("Invalid index")
            return None
        return self.dataset[idx]["width"], self.dataset[idx]["height"]
    
    def get_K(self, idx):
        """
            Get the K matrix.
        """
        if idx < 0 or idx >= self.nb_images:
            print("Invalid index")
            return None
        return np.asarray(self.dataset[idx]["K"])


    def get_Rt(self, idx):
        """
            Get the extrinsic parameters.
        """
        if idx < 0 or idx >= self.nb_images:
            print("Invalid index")
            return None
        R = np.asarray(self.dataset[idx]["R"])
        t = np.asarray(self.dataset[idx]["t"])
        return np.hstack((R, t.reshape((3, 1))))


    def get_rgb_filename(self, idx):
        """
            Get the rgb image filename.
        """
        if idx < 0 or idx >= self.nb_images:
            print("Invalid index")
            return None
        return self.dataset[idx]["file_name"]