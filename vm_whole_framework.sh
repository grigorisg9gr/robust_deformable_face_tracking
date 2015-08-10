#!/bin/bash

S1_DL='1_dlib_pred'
S1_DL_BB='1_dlib_detect'
#S2_DPM='2_dpm' # do not change this (otherwise, change manually in matlab
#S3_DL='3_dlib_pred'
S3_DL='3_ffld_ln' 
S4_AAM='4_fit_pbaam_eyes'
S5_SVM='5_svm_faces_eyes'
S6_AAM='6_fit_pbaam'
S7_SVM='7_svm_faces'

PYTHON_P='/vol/atlas/homes/grigoris/installation_folder/miniconda/envs/my_menpofast_nomkl/bin/python'   # path of python environment

if [[ -z $1 ]]; then
    PATH1='/vol/atlas/homes/grigoris/company_videos/competition/grigoris_1/'
else
    PATH1=$1
fi
if [ ! -d $PATH1 ]; then 						# confirm that the directory exists 
    echo 'The directory does not exist, the program is existing.'
    exit -1
fi

function call_framework(){
echo 'subprocess called'
echo $PATH1
  python dlib_predictor.py $PATH1 $S1_DL_BB $S1_DL
#  matlab -nodisplay -r 'cd /vol/atlas/homes/grigoris/external/dpm_matlab/voc-dpm/;run_with_override('\'$PATH1\'');exit()'
  #python dlib_predictor-predict_from_bb.py $PATH1  
  python ffld2.py $PATH1
  $PYTHON_P ps_pbaam.py $PATH1  $S3_DL $S4_AAM
  python  svm_isface.py $PATH1 $S4_AAM $S5_SVM
  $PYTHON_P loop_ps_pbaam.py $PATH1  $S5_SVM $S4_AAM $S6_AAM
  python  svm_isface.py $PATH1 $S6_AAM $S7_SVM
#  $PYTHON_P visualisations_to_videos.py $PATH1 
  #python visualise_landmarks.py $PATH1 c $S7_SVM $S1_DL 
  python visualise_landmarks.py $PATH1 g $S7_SVM
  #date  > date_whole_framework
  echo 'finished'
  $PYTHON_P completion_mail.py  $PATH1 $S1_DL $S2_DPM $S4_AAM $S5_SVM $S6_AAM $S7_SVM
}


call_framework #&  # way to fork a new process that can run independently afterwardsi
echo $PATH1
sleep 2
echo 'well going now, the child process must have run'



