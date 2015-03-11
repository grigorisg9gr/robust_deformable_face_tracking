#!/bin/bash

S1_DL='1_dlib_pred'
S1_DL_BB='1_dlib_detect'
S2_DPM='2_dpm' # do not change this (otherwise, change manually in matlab
S3_DL='3_dlib_pred'
S4_AAM='4_fit_pbaam'
S5_SVM='5_svm_faces'
S6_AAM='6_fit_pbaam'

#CODE_P='/homes/gchrysos/Documents/menpo-notebooks-0.4.0a1/notebooks/grigoris_tracker/automatic_scripts/v1.2/'
PYTHON_P='/vol/atlas/homes/grigoris/installation_folder/miniconda/envs/menpo/bin/python'   # path of python environment

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
#  $PYTHON_P dlib_predictor.py $PATH1 $S1_DL $S1_DL_BB
#  ## matlab -nodisplay -r 'cd /vol/atlas/homes/grigoris/external/dpm_matlab/voc-dpm/;startup; train_dpm_for_all_clips_in_folder();exit()'  #old version
  matlab -nodisplay -r 'cd /vol/atlas/homes/grigoris/external/dpm_matlab/voc-dpm/;run_with_override('\'$PATH1\'');exit()'
#  $PYTHON_P dlib_predictor-predict_from_bb.py $PATH1  
  #echo 'calling landmark prediction from bb'
  #$PYTHON_P pb_aam_correct_missed_frames_automatically_whole_folder.py $PATH1  $S1_DL $S4_AAM
  #echo 'function to make part-based AAM for landmark correction'
  #$PYTHON_P isface_svm_training_and_testing_fast_whole_folder.py $PATH1 $S1_DL $S4_AAM $S5_SVM
  #echo 'train svm and detect wrongly localised faces'
#  $PYTHON_P 2_pb_aam_correct_missed_frames_automatically_whole_folder.py $PATH1  $S5_SVM $S6_AAM
#  $PYTHON_P visualisations_to_videos.py $PATH1 
  echo 'converting to videos all frames written during the different steps' 
  #date  > date_whole_framework
  echo 'finished'
  $PYTHON_P completion_mail.py  $PATH1
}


call_framework # way to fork a new process that can run independently afterwardsi
echo $PATH1
sleep 2
echo 'well going now, the child process must have run'
#PATH1='/vol/atlas/homes/grigoris/company_videos/competition/over_90mb_video/'
#PATH1='/vol/atlas/homes/grigoris/company_videos/competition/cut_initial/'
#python $CODE_P/dlib_predictor.py $PATH1 $S1_DL $S1_DL_BB
#  matlab -nodisplay -r 'cd /vol/atlas/homes/grigoris/external/dpm_matlab/voc-dpm/;run_with_override('\'$PATH1\'');exit()'
#  python dlib_predictor-predict_from_bb.py $PATH1  
#  echo 'calling landmark prediction from bb'
#PATH1='/vol/atlas/homes/grigoris/company_videos/competition/over_90mb_video/'
#echo 'subprocess called'
#echo $PATH1
  #python dlib_predictor.py $PATH1
#  ## matlab -nodisplay -r 'cd /vol/atlas/homes/grigoris/external/dpm_matlab/voc-dpm/;startup; train_dpm_for_all_clips_in_folder();exit()'  #old version
#  matlab -nodisplay -r 'cd /vol/atlas/homes/grigoris/external/dpm_matlab/voc-dpm/;run_with_override('\'$PATH1\'');exit()'




