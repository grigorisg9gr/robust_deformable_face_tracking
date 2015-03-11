
# coding: utf-8

# In[1]:

# grigoris, 5/12: This function is used to accept two set of landmarks and the respective frames and 
# print the frames with overlaid both groups of landmarks
# grigoris, 22/1: Update in order to read all landmarks per image (if there are more faces detected)


import matplotlib as mpl           # significant feature: For using the savefig in the python terminal. Should be added
mpl.use('Agg')                     # in the beginning of the program. http://stackoverflow.com/a/4935945/1716869
import os, sys
import menpo.io as mio
from utils import (mkdir_p, check_if_path)
from utils.pipeline_aux import (check_img_type)
from utils.path_and_folder_definition import *  # import paths for databases, folders and visualisation options

multiple_faces_per_image = False     # If set to false, then in each image can have at most one face (landmark group),
# which will have the same name as the image.
img_type_out = '.png'

output_height = 640
output_width = 640


if __name__ == '__main__':
    args = len(sys.argv)
    if args > 1:
        path_0 = str(sys.argv[1])
    else:
        raise RuntimeError('file not called with initial path')
    if args > 3:
        ln1 = str(sys.argv[2])
        ln2 = str(sys.argv[3])
    else:
        ln1 = '1_dlib_pred'
        ln2 = '3_dlib_pred'


def import_landm(group_in, file1, im):
    try:
        ln = mio.import_landmark_file(file1)
        im.landmarks[group_in] = ln
        return 1
    except ValueError:
        return 0

import glob
def call_import_landm(group_in, file1, im):
    res = glob.glob(file1)
    if len(res) == 0:
        return 0
    return import_landm(group_in, res[0], im)


def search_and_view_landm(ln_n, path1, im, name, color, viewer=[]):
    res = glob.glob(path1 + name)
    if len(res) == 0: 
        return import_landm(ln_n, path1 + name, im), viewer
    else:
        for kk in res:
            im.landmarks.clear()
            ok = import_landm(ln_n, kk, im)
            viewer = im.view_landmarks(group=ln_n, render_numbering=False, lmark_view_kwargs={'colours': color}) 
        return ok, viewer    


# colour = ['r','b','g',[.2,.4,.8],[.2,.4,.2], [.8,.4,.8],[.6,.0,.3]]; col_len = len(colour) ##################################### TEMP!!!! Overwrite the general options

def compare_landmarks_frame(fr_name, frames_path, ln1, ln2, p1, p2, path_all_clip):
    if fr_name[-4::]!=img_type: return # in case they are something different than an image
    rand = randint(1,10000);  plt.figure(rand)
    im = mio.import_image(frames_path + fr_name, normalise=False) 
    name = fr_name[:-4] + '*.pts'
    if multiple_faces_per_image == False:
        # code below, the case where there is maximum one face/group of landmarks per image

        ln1_1 = call_import_landm(ln1, p1 + name, im)
        ln2_1 = call_import_landm(ln2, p2 + name, im)

        if ln1_1 == 1 and ln2_1 == 1:
            im.crop_to_landmarks_proportion_inplace(0.3, group=ln2)
            # min_indices = [im.landmarks[ln2][None].points[30, 0] - (output_height / 2), im.landmarks[ln2][None].points[30, 1] - (output_width / 2)]
            # max_indices = [im.landmarks[ln2][None].points[30, 0] + (output_height / 2), im.landmarks[ln2][None].points[30, 1] + (output_width / 2)]
            # im.crop_inplace(0.3, group=ln2)
            # viewer = im.view_landmarks(group=ln1, render_numbering=False, lmark_view_kwargs={'colours': red}) #, avoid "new_figure=True"
            # viewer = im.view_landmarks(group=ln2, render_numbering=False, lmark_view_kwargs={'colours': blue}) #, avoid "new_figure=True" -> assigns new id
            ln1_centre = im.landmarks[ln1].lms.centre()
            ln2_centre = im.landmarks[ln2].lms.centre()
            if (abs(ln1_centre[0] - ln2_centre[0]) < 150 and abs(ln1_centre[1] - ln2_centre[1]) < 150):
                viewer = im.view_landmarks(group=ln1, render_numbering=False,  marker_face_colour=colour[0], marker_edge_colour=colour[0])
                viewer = im.view_landmarks(group=ln2, render_numbering=False,  marker_face_colour=colour[1], marker_edge_colour=colour[1])
            else:
                viewer = im.view_landmarks(group=ln2, render_numbering=False,  marker_face_colour=colour[1], marker_edge_colour=colour[1])
        elif ln2_1 == 1:
            im.crop_to_landmarks_proportion_inplace(0.3, group=ln2)
            # viewer = im.view_landmarks(group=ln2, render_numbering=False, lmark_view_kwargs={'colours': blue}) #, avoid "new_figure=True" -> assigns new id
            viewer = im.view_landmarks(group=ln2, render_numbering=False,  marker_face_colour=colour[1], marker_edge_colour=colour[1])
        elif ln1_1 == 1:
            im.crop_to_landmarks_proportion_inplace(0.3, group=ln1)
            # viewer = im.view_landmarks(group=ln1, render_numbering=False, lmark_view_kwargs={'colours': red}) #, avoid "new_figure=True"
            viewer = im.view_landmarks(group=ln1, render_numbering=False,  marker_face_colour=colour[0], marker_edge_colour=colour[0])
        else:
            viewer = im.view()
    else:    
        ln1_1, viewer = search_and_view_landm(ln1, p1, im, name, red)
        ln2_1, viewer = search_and_view_landm(ln2, p2, im, name, blue, viewer)
        if ln1_1 == 0 and ln2_1 == 0: 
            viewer = im.view()
    # viewer.figure.savefig(path_all_clip + fr_name)
    viewer.save_figure(path_all_clip + fr_name[:-4] + img_type_out, pad_inches=0., overwrite=True, format=img_type_out[1::])
    plt.close(rand)


# VARIATION THAT WRITES ONLY WHEN BOTH LANDMARK POINTS EXIST: 
# def compare_landmarks_frame(fr_name, frames_path, ln1,ln2, p1, p2, path_all_clip):
#     rand = randint(1,10000);  plt.figure(rand)
#     im = mio.import_image(frames_path + fr_name, normalise=False) 
#     name = fr_name[:-4] + '.pts';
#     ln1_1 = import_landm(ln1, p1 + name, im)
#     ln2_1 = import_landm(ln2, p2 + name, im)
#     if ln1_1 == 1 and  ln2_1 == 1:
#         viewer = im.view_landmarks(group=ln1, render_numbering=False, lmark_view_kwargs={'colours': red}) #, avoid "new_figure=True" -> assigns new id
#         viewer = im.view_landmarks(group=ln2, render_numbering=False, lmark_view_kwargs={'colours': blue}) #, avoid "new_figure=True" -> assigns new id
#         viewer.figure.savefig(path_all_clip + fr_name)
#     plt.close(rand)



import matplotlib.pyplot as plt
import numpy as np
from random import randint
from joblib import Parallel, delayed


def process_clip(clip_name, ln1, ln2):
    path_land_1 = path_0 + ln1 + '/'
    path_land_2 = path_0 + ln2 + '/'
    p1 = path_land_1 + clip_name + '/'; p2 = path_land_2 + clip_name + '/'
    print p1, p2
    if not(os.path.isdir(p1)): print('Skipped clip (ln1) ' + clip_name);return
    #if not(os.path.isdir(p2)): print('Skipped clip (ln2) ' + clip_name);return
    path_comp = path_0 + foldvis + foldcmp; mkdir_p(path_comp);
    path_all_clip_0 = path_comp + ln1 + '_' + ln2 + '/'; mkdir_p(path_all_clip_0)
    path_all_clip = path_all_clip_0 + clip_name + '/'; mkdir_p(path_all_clip)
    frames_path = path_clips + clip_name + '/'
    list_frames = sorted(os.listdir(frames_path));
    Parallel(n_jobs=-1, verbose=4)(delayed(compare_landmarks_frame)
                                   (fr_name, frames_path, ln1, ln2, p1, p2, path_all_clip) for fr_name in list_frames)
    # [compare_landmarks_frame(fr_name, frames_path, ln1,ln2, p1, p2, path_all_clip) for fr_name in list_frames]


path_clips = path_0 + frames


img_type = '.png'
# blue = np.array([[0], [0], [1]])                                 # fix the color of the viewer (landmark points)
# red  = np.array([[1], [0], [0]])

if not check_if_path(path_clips, 'The path (%s) does not exist, skip the comparison' % path_clips):
    exit()
list_clips = sorted(os.listdir(path_clips))
for clip in list_clips:
    process_clip(clip, ln1 = ln1, ln2 = ln2)

