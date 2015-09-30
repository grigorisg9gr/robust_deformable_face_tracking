import os
import sys
from utils.pipeline_aux import check_initial_path
from dlib_predictor import main_for_generic_detector
from ffld2 import main_for_ps_detector
from ps_pbaam import main_for_ps_aam

s = os.path.sep  # OS separator

#######################################################
#######################################################
##############   folders for landmarks   ##############
s_1_det = '1_detect' + s
s_1_pred = '1_pred' + s
s_2_psd = '' + s
s_3_ln = '3_ffld_ln' + s
s_4_pbaam = '4_pbaam' + s
s_5_svm = '5_svm' + s
s_6_pbaam = '6_pbaam' + s
s_7_svm = '7_svm_faces' + s
#######################################################
#######################################################


def run_main(path_clips):
    main_for_generic_detector(path_clips, s_1_det, s_1_pred)
    main_for_ps_detector(path_clips, s_1_det, s_2_psd, s_2_psd[:-1] + '_models' + s, s_3_ln)
    main_for_ps_aam(path_clips, s_3_ln, s_4_pbaam, s_4_pbaam[:-1] + '_models' + s,
                    n_shape=[3, 12], n_appearance=[50, 100])
    # TODO: Add the rest of the steps. Not finished yet. 


if __name__ == '__main__':
    args = len(sys.argv)
    path_clips_m = check_initial_path(args, sys.argv)
    run_main(path_clips_m)