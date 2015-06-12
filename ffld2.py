
import matplotlib as mpl           # significant feature: For using the savefig in the python terminal. Should be added
mpl.use('Agg')                     # in the beginning of the program. http://stackoverflow.com/a/4935945/1716869
import os
import sys
import menpo.io as mio
from menpo.feature import no_op
feat = no_op

# path_clips = '/vol/atlas/homes/grigoris/company_videos/competition/grigoris_3/';
from utils import (mkdir_p, print_fancy)
from utils.pipeline_aux import (check_img_type, check_path_and_landmarks, load_images)
from utils.path_and_folder_definition import *  # import paths for databases, folders and visualisation options

if __name__ == '__main__':
    args = len(sys.argv)
    if args > 1:
        path_clips = str(sys.argv[1])
        if not(os.path.isdir(path_clips)):
            raise RuntimeError('The path %s does not exist as base folder' % path_clips)
    else:
        raise RuntimeError('file not called with initial path')

    if 2 < args < 7: #args > 2 and args < 5:
        in_bb_fol = str(sys.argv[2]) + '/'
        out_bb_fol = str(sys.argv[3]) + '/'
        out_model_fol = str(sys.argv[4]) + '/'
        out_landmarks_fol = str(sys.argv[5]) + '/'
        print in_bb_fol, '   ', out_landmarks_fol
    else:
        in_bb_fol = '1_dlib_detect/'
        out_bb_fol = '2_ffld/'
        out_model_fol = '2_ffld_models/'
        out_landmarks_fol = '3_ffld_ln/'


p_det_0 = path_clips + out_bb_fol #; mkdir_p(p_det_0)
p_det_1 = path_clips + out_landmarks_fol #; mkdir_p(p_det_1)
p_det_bb_0 = path_clips + in_bb_fol  # existing bbox of detection
p_save_model = path_clips + out_model_fol; mkdir_p(p_save_model) # path that trained models will be saved


import numpy as np
from menpo.shape import PointCloud
from joblib import Parallel, delayed
import glob

from menpodetect.ffld2 import FFLD2Detector, load_ffld2_frontal_face_detector, train_ffld2_detector
from menpodetect.dlib.conversion import pointgraph_to_rect
import dlib

predictor_dlib = dlib.shape_predictor('/vol/atlas/homes/grigoris/raps_menpo/shape_predictor_68_face_landmarks.dat')

def detection_to_pointgraph(detection):
    return PointCloud(np.array([(p.y, p.x) for p in detection.parts()]))

    
def predict_in_frame(frame_name, frames_path, p_det_landm, p_det_bb):
    global detector
    if frame_name[-4:]!=img_type:
        return
    im = mio.import_image(frames_path + frame_name, normalise=True)
    if im.n_channels == 3: im = im.as_greyscale(mode='luminosity')
#     print frame_name
    res_dlib = detector(im)
    num_res = len(res_dlib)
    if num_res == 0:
        return 
    else:
        num1 = 1                # num1 and s1: Values if there are more than 10 detections in the image
        if num_res > 9:
            num1 = 2
        s1 = '%0' + str(num1)
        im_pili = im.as_PILImage()
        for kk in range(0,1):   # num_res to keep all, here keeping ONLY the most confident one
            pts_end = '_' + str(kk) + '.pts'
            mio.export_landmark_file(im.landmarks['ffld2_' + (s1 + 'd')%kk], p_det_bb + frame_name[:-4] + pts_end, overwrite=True)
            # convert to landmarks
            det_frame = predictor_dlib(np.array(im_pili), pointgraph_to_rect(im.landmarks['ffld2_' + (s1 + 'd')%kk].lms))
            init_pc = detection_to_pointgraph(det_frame)
            group_kk = 'bb_' + str(kk)
            im.landmarks[group_kk] = init_pc
            mio.export_landmark_file(im.landmarks[group_kk], p_det_landm + frame_name[:-4] + pts_end, overwrite=True)



negative_images = [i.as_greyscale(mode='channel', channel=1) for i in mio.import_images('/vol/atlas/homes/pts08/non_person_images',normalise=False, max_images=200)] #200
detector = []
def process_clip(clip_name):
    print clip_name
    frames_path = path_clips + frames + clip_name + '/'
    # if not(os.path.isdir(p_det_bb_read)):
    #     print('Skipped clip ' + clip_name + ' because it does not have previous landmarks (detect)')
    #     return
    if not check_path_and_landmarks(frames_path, clip_name, p_det_bb_0 + clip_name + '/'): # check that paths, landmarks exist
        return

    list_frames = sorted(os.listdir(frames_path))
    # build the detector
    global detector
    training_pos = load_images(list_frames, frames_path, p_det_bb_0, clip_name, max_images=300)
    if len(training_pos) == 0:
        print('No positives found for the clip %s, skipping it.' % clip_name)
        return
    print len(training_pos), training_pos[0].shape
    ps_model = train_ffld2_detector(training_pos, negative_images, n_components=1, n_relabel=4)
    ps_model.save(p_save_model + clip_name + '.model')
    detector = FFLD2Detector(ps_model)

    p_det_bb = p_det_0 + clip_name + '/'; mkdir_p(p_det_bb)
    p_det_landm = p_det_1 + clip_name + '/'; mkdir_p(p_det_landm)

    [predict_in_frame(frame_name, frames_path, p_det_landm, p_det_bb) for frame_name in list_frames]

print_fancy('Training person specific model with FFLD')
list_clips = sorted(os.listdir(path_clips + frames))
img_type = check_img_type(list_clips, path_clips + frames)  # assumption that all clips have the same extension, otherwise run in the loop for each clip separately.
[process_clip(clip_name) for clip_name in list_clips];


