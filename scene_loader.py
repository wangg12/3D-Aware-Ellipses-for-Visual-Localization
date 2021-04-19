import json
import numpy as np
from ellcv.types import Ellipsoid

class Scene_loader:
    """
        Interace with our own scene file format.
    """

    def __init__(self, filename):
        with open(filename, "r") as fin:
            self.scene = json.load(fin)

        self.category_id_to_label_map = self.scene["category_id_to_label"]
        self.objects = []
        self.object_id_map = {}
        for obj in self.scene["objects"]:
            obj_data = {
                "category_id": obj["category_id"],
                "object_id": obj["object_id"],
                "ellipsoid": Ellipsoid.from_dict(obj["ellipsoid"])
            }
            self.object_id_map[obj["object_id"]] = len(self.objects)
            self.objects.append(obj_data)

    def __len__(self):
        return len(self.objects)

    def get_object(self, idx):
        """
            Get an object.
        """
        if idx < 0 or idx >= len(self.objects):
            print("Invalid index")
            return None
        return self.objects[idx]

    def get_object_by_id(self, object_id):
        if idx not in self.object_id_map.keys():
            print("Invalid object id")
            return None
        return self.objects[self.object_id_map[object_id]]

    def get_category_label(self, cat):
        if cat not in self.category_id_to_label_map.keys():
            print("Invalid category id")
            return None
        return self.category_id_to_label_map[cat]

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < 0 or self._index >= len(self.objects):
            raise StopIteration()
        cur_obj = self.objects[self._index]
        self._index += 1
        return cur_obj

    def __getitem__(self, key):
        return self.objects[key]
