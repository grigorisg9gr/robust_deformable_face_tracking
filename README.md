
To use the code you can call the whole pipeline from one of the bash scripts.
Probably you need to adapt the paths in `utils/path_and_folder_definition.py` to fit your system.

Most of the python files expect to be provided a base path. Inside that base path a `frames/` folder 
is expected with the frames of the respective clip.
In the `example_folder_structure/` the structure of a folder that the code should be called can be found. That is, you can put the clip folder you want in the folder `frames/` and inside the clip folder to put the respective clips.
The scripts will iteratively run for all clips in the `frames/` folder.
The script will produce and save the results in folders that will be created in the `example_folder_structure/` path. 

Explanation of the files:

* dlib_predictor.py: Uses pre-trained (generic) detector to detect the bounding box of the face and then applies a shape predictor/regressor (from dlib).
* ffld2.py: Trains a person specific detector. It utilises the input bounding boxes of the previous step to train the model and then detects using that model. Additionally, it applies the same shape predictor as in dlib_predictor.py.
* ps_pbaam.py: Trains a generic + ps part-based AAM and then fits it on the person.
* svm_is_face.py: Checks whether the landmarks correspond to an actual human face. If the trained SVM does not exist (pickle file), it trains it.


