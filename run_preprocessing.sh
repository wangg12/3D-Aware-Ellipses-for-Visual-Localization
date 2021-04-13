
python preprocess_7-Scenes.py -i $1/seq-01 $1/seq-04 $1/seq-06 -o 7-Scenes_Chess_dataset_train.json
python preprocess_7-Scenes.py -i $1/seq-02 $1/seq-03 $1/seq-05 -o 7-Scenes_Chess_dataset_test.json

python annotate_objects.py scene.json 7-Scenes_Chess_dataset_train.json 7-Scenes_Chess_dataset_train_with_obj_annot.json
python annotate_objects.py scene.json 7-Scenes_Chess_dataset_test.json  7-Scenes_Chess_dataset_test_with_obj_annot.json
