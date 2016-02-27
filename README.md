
This is the code accompanying the paper [`Offline Deformable Face Tracking in Arbitrary Videos`, ICCV 2015 Workshops.](http://ibug.doc.ic.ac.uk/media/uploads/documents/shen_the_first_facial_iccv_2015_paper.pdf)
To use the code you can call the whole pipeline by calling the run_pipeline.py with the path of the videos.
Probably you need to adapt the paths in `utils/path_and_folder_definition.py` to fit your system.

#### **Folder structure**
The pipeline proposed assumes the following folder structure:

```
path_base
    │
    └───frames
            │
            └───[name_of_clip]
                    │ [frame_name].[extension] (e.g. 000001.png)
                    │ [frame_name].[extension] (e.g. 000002.png)
                    │ ...
            │ ...
    │
    └───gt_landmarks  (not required for executing the code)
            │
            └───[name_of_clip]
                    │ [file_name].[extension] (e.g. 000001.pts)
                    │ ...
            │ ...
```

In the `example_folder_structure/` the structure as described above can be found. 
The scripts will iteratively run for all clips in the `frames/` folder. 


Explanation of the different pipeline files:

* dlib_predictor.py: Uses pre-trained (generic) detector to detect the bounding box of the face and then applies a shape predictor/regressor (from dlib).
* ffld2.py: Trains a person specific detector. It utilises the input bounding boxes of the previous step to train the model and then detects using that model. Additionally, it applies the same shape predictor as in dlib_predictor.py.
* ps_pbaam.py: Trains a generic + adaptive part-based AAM and then fits it on the person.

#### **Dependency**
Apart from menpo [(menpo, menpodetect, menpofit)](https://github.com/menpo/menpo) the following packages are used:
* joblib ``` pip install joblib```
* research_pyutils from [this repository](https://github.com/grigorisg9gr/pyutils).

Additionally, for the predictor in the dlib_predictor.py, you need to provide a pre-trained model for landmark localisation. 

#### **Feedback**
If you do have any questions or improvements, feel free to open issues here or contribute right away. Feedback is always appreciated.

#### **License**
Apache 2, see LICENSE file for details.
