
import matplotlib as mpl           # significant feature: For using the savefig in the python terminal. Should be added
mpl.use('Agg')                     # in the beginning of the program. http://stackoverflow.com/a/4935945/1716869
import os, sys
import menpo
import menpo.io as mio
from utils import (mkdir_p, check_if_path)
from utils.path_and_folder_definition import *  # import paths for databases, folders and visualisation options
from utils.pipeline_aux import check_img_type

if __name__ == '__main__':
    args = len(sys.argv)
    if args>1:
        path_clips = str(sys.argv[1])
        if not(os.path.isdir(path_clips)):
            raise RuntimeError('The path %s does not exist as base folder' % path_clips)
    else:
        raise RuntimeError('file not called with initial path')

    if args > 2 and args < 5:
        in_bb_fol = str(sys.argv[2]) + '/'
        out_landmarks_fol = str(sys.argv[3]) + '/'
        print out_landmarks_fol, '   ', in_bb_fol
    else:
        in_bb_fol = '2_dpm/'
        out_landmarks_fol = '3_dlib_pred/'

#path_clips = '/vol/atlas/homes/grigoris/company_videos/interesting_one_person_videos/';


# load dlib point predictor
import dlib
from menpodetect.dlib.conversion import pointgraph_to_rect
from menpo.shape import PointCloud
import numpy as np

def detection_to_pointgraph(detection):
    return PointCloud(np.array([(p.y, p.x) for p in detection.parts()]))

predictor_dlib = dlib.shape_predictor('/vol/atlas/homes/grigoris/raps_menpo/shape_predictor_68_face_landmarks.dat')


# definition of paths
if visual == 1:
    p_det_0_0 = path_clips + foldvis; mkdir_p(p_det_0_0)
    p_det_0 = p_det_0_0 + out_landmarks_fol; mkdir_p(p_det_0)
p_det_1 = path_clips + out_landmarks_fol; mkdir_p(p_det_1)
p_det_bb_0 = path_clips + in_bb_fol # existing bbox of detection


import matplotlib.pyplot as plt
import numpy as np
from random import randint
from menpo.shape import PointCloud
from joblib import Parallel, delayed
import glob


def return_only_biggest_bbox(im, group_out='object_biggest', delete_rest=True):
    # accepts an image with several label_groups in the form of a bounding box and returns the one with the biggest area. 
    max_area = 0; label_keep = []
    for label1 in im.landmarks.group_labels:
        area1 = bbox_area(im.landmarks[label1][None].points)
        if area1 > max_area: 
            max_area = area1
            label_keep = label1
    print label_keep
    if delete_rest:
        pts_keep = im.landmarks[label_keep][None].points
        for label1 in im.landmarks.group_labels:
            del im.landmarks[label1]
    im.landmarks[group_out] = PointCloud(pts_keep).bounding_box()
      

def bbox_area(bbox): 
    # bbox in the form of menpo landmarks, with points starting from left top corner and going anti-clockwise. 
    return (bbox[1][0] - bbox[0][0])*(bbox[2][1] - bbox[0][1])

    
def predict_in_frame(frame_name, frames_path, p_det_landm, p_det_vis, p_det_bb):
    if frame_name[-4::]!=img_type:
        return # in case they are something different than an image
    if visual==1: rand = randint(1,10000);  plt.figure(rand)
    im = mio.import_image(frames_path + frame_name, normalise=True)
    if im.n_channels == 3: im = im.as_greyscale(mode='luminosity')
#     res_dlib = dlib_init_detector(im, group_prefix='dlib'); num_res = len(res_dlib)
    res = glob.glob(p_det_bb + frame_name[:-4] + '*.pts')
    if len(res) ==0 and (visual == 1):
        viewer = im.view()
        # viewer.figure.savefig(p_det_vis + frame_name[:-4] + img_type)
        viewer.figure.savefig(p_det_vis + frame_name[:-4] + img_type)
        plt.close(rand)
    elif len(res)>0:
#         num1 = 1;                # num1 and s1: Values if there are more than 10 detections in the image
#         if len(res)>9: num1 = 2; 
#         s1 = '%0' + str(num1)
        im_pili = im.as_PILImage()
        for kk in range(0,len(res)):
            # load bbox 
            ln = mio.import_landmark_file(res[kk])
            im.landmarks['dlib_' + str(kk)] = ln
            # apply predictor
            pts_end = '_' + str(kk) + '.pts'
            try: 
                det_frame = predictor_dlib(np.array(im_pili), pointgraph_to_rect(im.landmarks['dlib_' + str(kk)].lms)) 
            except:
                print 'Missed: ' + frames_path + frame_name
                if visual==1:
                    viewer = im.view();
                    viewer.figure.savefig(p_det_vis + frame_name[:-4] + img_type)
                    plt.close(rand)
                return
            init_pc = detection_to_pointgraph(det_frame)
            group_kk = 'bb_' + str(kk)
            im.landmarks[group_kk] = init_pc
            # export and print result
            mio.export_landmark_file(im.landmarks[group_kk], p_det_landm + frame_name[:-4] + pts_end, overwrite=True)
            if visual==1:
                _c1 = colour[kk%col_len]
                if len(res)==1: im.crop_to_landmarks_proportion_inplace(0.3)   ## zoom-in to see finer details for fitting, remove if unnecessary
                viewer = im.view_landmarks(group=group_kk, render_numbering=False,  marker_face_colour=_c1, marker_edge_colour=_c1)
                # viewer = im.view_landmarks(group=group_kk, render_numbering=False, lmark_view_kwargs={'colours': colour[kk%col_len]})
        if visual==1:
            viewer.save_figure(p_det_vis + frame_name[:-4] + img_type, pad_inches=0., overwrite=True, format=img_type[1::])
            # viewer.figure.savefig(p_det_vis + frame_name[:-4] + img_type)
            plt.close(rand) # plt.close('all') #problem with parallel # http://stackoverflow.com/a/21884375/1716869
  


def process_clip(clip_name):
    p_det_bb = p_det_bb_0 + clip_name + '/'                      #### load bbox of detection
    if not(os.path.isdir(p_det_bb)):
        print('Skipped clip ' + clip_name + ' because it does not have previous landmarks (from dpm)')
        return
    print(clip_name)
    if visual == 1:
        p_det_vis = p_det_0 + clip_name + '/' ; mkdir_p(p_det_vis)
    else:
        p_det_vis = ''
    p_det_landm = p_det_1 + clip_name + '/'; mkdir_p(p_det_landm)
    frames_path = path_clips + frames + clip_name + '/'
    list_frames = sorted(os.listdir(frames_path));
    Parallel(n_jobs=-1, verbose=4)(delayed(predict_in_frame)
                    (frame_name, frames_path, p_det_landm, p_det_vis, p_det_bb) for frame_name in list_frames);

#     [predict_in_frame(frame_name, frames_path, p_det_landm, p_det_vis, p_det_bb) for frame_name in list_frames]



# iterates over all clips in the folder and calls sequentially the function process_clip
list_clips = sorted(os.listdir(path_clips + frames))
img_type = check_img_type(list_clips, path_clips + frames)  # assumption that all clips have the same extension, otherwise run in the loop for each clip separately.
[process_clip(clip_name) for clip_name in list_clips]


