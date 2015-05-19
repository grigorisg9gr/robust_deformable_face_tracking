
import matplotlib as mpl           # significant feature: For using the savefig in the python terminal. Should be added
mpl.use('Agg')                     # in the beginning of the program. http://stackoverflow.com/a/4935945/1716869
import os
import sys
import menpo.io as mio
from utils import (mkdir_p, check_if_path)
from utils.path_and_folder_definition import *  # import paths for databases, folders and visualisation options
from utils.pipeline_aux import check_img_type

if __name__ == '__main__':
    args = len(sys.argv)
    if args > 1:
        path_clips = str(sys.argv[1])
        if not(os.path.isdir(path_clips)):
            raise RuntimeError('The path %s does not exist as base folder' % path_clips)
    else:
        raise RuntimeError('file not called with initial path')

    if 2 < args < 5:
        out_landmarks_fol = str(sys.argv[3]) + '/'
        out_bb_fol = str(sys.argv[2]) + '/'
        print out_landmarks_fol, '   ', out_bb_fol
    else:
        out_landmarks_fol = '1_dlib_pred/'
        out_bb_fol = '1_dlib_detect/'


# definition of paths
p_det_1 = path_clips + out_landmarks_fol
p_det_bb_0 = path_clips + out_bb_fol       #### save bbox of detection


# load dlib detector and point predictor
import dlib
from menpodetect.dlib.conversion import pointgraph_to_rect
from menpodetect import load_dlib_frontal_face_detector
from menpo.shape import PointCloud
import numpy as np
from joblib import Parallel, delayed

def detection_to_pointgraph(detection):
    return PointCloud(np.array([(p.y, p.x) for p in detection.parts()]))

dlib_init_detector = load_dlib_frontal_face_detector()
predictor_dlib = dlib.shape_predictor(path_shape_pred)


def detect_in_frame(frame_name, frames_path, p_det_landm, p_det_bb):
    if frame_name[-4:] != img_type:
        return # in case they are something different than an image
    im = mio.import_image(frames_path + frame_name, normalise=True)
    if im.n_channels == 3: im = im.as_greyscale(mode='luminosity')
    # call dlib detector
    res_dlib = dlib_init_detector(im, group_prefix='dlib')
    num_res = len(res_dlib)
    if num_res > 0:
        num1 = 1                # num1 and s1: Values if there are more than 10 detections in the image
        if num_res>9: num1 = 2
        s1 = '%0' + str(num1)
        im_pili = im.as_PILImage()
        for kk in range(0,num_res):
            pts_end = '_' + str(kk) + pts_type_out # define the ending of each pts that will be exported
            mio.export_landmark_file(im.landmarks['dlib_' + (s1 + 'd')%kk], p_det_bb + frame_name[:-4] + pts_end, overwrite=True)
            # from bounding box to points (dlib predictor)
            det_frame = predictor_dlib(np.array(im_pili), pointgraph_to_rect(im.landmarks['dlib_' + (s1 + 'd')%kk].lms))
            init_pc = detection_to_pointgraph(det_frame)
            group_kk = 'bb_' + str(kk)
            im.landmarks[group_kk] = init_pc
            mio.export_landmark_file(im.landmarks[group_kk], p_det_landm + frame_name[:-4] + pts_end, overwrite=True)



def process_clip(clip_name):
    """
    Function that processes one clip. It creates the essential paths (for landmarks and visualisations)
    and then calls the function detect_in_frame for each frame of the clip.
    """
    frames_path = path_clips + frames + clip_name + '/'
    list_frames = sorted(os.listdir(frames_path))
    if not check_if_path(frames_path, 'Skipped clip ' + clip_name + ' because its path of frames is not valid'):
        return
    print(clip_name)
    p_det_landm = p_det_1 + clip_name + '/'; mkdir_p(p_det_landm)
    p_det_bb = p_det_bb_0 + clip_name + '/'; mkdir_p(p_det_bb) #### save bbox of detection

    Parallel(n_jobs=-1, verbose=4)(delayed(detect_in_frame)
                    (frame_name, frames_path, p_det_landm, p_det_bb) for frame_name in list_frames);



# iterates over all clips in the folder and calls sequentially the function process_clip
list_clips = sorted(os.listdir(path_clips + frames))
img_type = check_img_type(list_clips, path_clips + frames)  # assumption that all clips have the same extension, otherwise run in the loop for each clip separately.
[process_clip(clip_name) for clip_name in list_clips
 if not(clip_name in list_done)]

