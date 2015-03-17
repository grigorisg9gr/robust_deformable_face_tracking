
multiple_faces_per_image = False     # If set to false, then in each image can have at most one face (landmark group),
# which will have the same name as the image.
img_type_out = '.png'


from utils import (mkdir_p, check_if_path)
from utils.pipeline_aux import check_path_and_landmarks
from utils.visualisation_aux import generate_frames_max_bbox
from utils.path_and_folder_definition import *  # import paths for databases, folders and visualisation options
import os, sys
import re

if __name__ == '__main__':
    args = len(sys.argv)
    if args > 1:
        path_clips = str(sys.argv[1])
        if not(os.path.isdir(path_clips)):
            raise RuntimeError('The path %s does not exist as base folder' % path_clips)
    else:
        raise RuntimeError('file not called with initial path')

    list_landm = []
    if args > 2: # all translated as paths of landmarks to be saved
        for i in range(2, args):
            list_landm.append(sys.argv[i])
    else:
        list_landm = ['5_svm_faces', '1_dlib_pred']


proportion = 0.2
figure_size = (10, 8)
overwrite = True
save_original = True
frames_format = '.png'

# prwto afora stis original eikones, deutero afora stis cropped
render_options = {'colours':    [colour,
                                 colour],
                  'sizes':      [[5, 5, 5, 5, 5, 5, 5],
                                 [10, 10, 10, 10, 10, 10, 10]],
                  'edgesizes':  [[1, 1, 1, 1, 1, 1, 1],
                                 [2, 2, 2, 2, 2, 2, 2]]}




# def for_each_landmark_folder(fold_landm):
#     save_path = save_path_0 + fold_landm + '/'; mkdir_p(save_path)
#     for clip in list_clips:
#         path_frames = path_f + clip + '/'
#         path_lndm = path_clips + fold_landm + '/' + clip + '/'
#         if not check_path_and_landmarks(path_frames, clip, path_lndm):
#             continue
#         print(clip)
#         save_path_i = save_path + clip + '/'; mkdir_p(save_path_i)
#         imgs = generate_frames_max_bbox(path_frames, frames_format, [path_lndm],
#                                         pts_format, [fold_landm],
#                                         save_path_i, proportion, figure_size, overwrite, save_original, render_options)


save_path_0 = path_clips + foldvis + '/'; mkdir_p(save_path_0)
save_path_1 = save_path_0 + foldcmp + '/'; mkdir_p(save_path_1)
pts_format = ['_0' + pts_type_out]
path_f = path_clips + frames
list_clips = sorted(os.listdir(path_f))

list_landm_f = []
_pattern = re.compile('[^a-zA-Z0-9]+')
for landm in list_landm:
    if not check_if_path(path_clips + landm + '/', ''):
        continue
    list_landm_f.append(landm)
if len(list_landm_f) == 0:
    print('No valid landmark folders found from the list.')
    exit(1)

name_fold = ''.join(map(lambda x: '%s_' % _pattern.sub('', x), list_landm_f))  # the name of the folder that the landmarks will be saved in
name_fold = name_fold[:-1] # stripping the last _
pts_format = pts_format * len(list_landm_f) # replicate list elements as many times as the different landmark groups
# print name_fold, list_landm_f, save_path_1 + name_fold
save_path_2 = save_path_1 + name_fold + '/'; mkdir_p(save_path_2)

for clip in list_clips:
    path_frames = path_f + clip + '/'
    path_landm = []
    for landm in list_landm_f:
        if not check_if_path(path_clips + landm + '/' + clip + '/', ''):
            continue
        path_landm.append(path_clips + landm + '/' + clip + '/')
    print(clip)
    save_path_i = save_path_2 + clip + '/'; mkdir_p(save_path_i)
    imgs = generate_frames_max_bbox(path_frames, frames_format, path_landm,
                                    pts_format, list_landm_f,
                                    save_path_i, proportion, figure_size, overwrite, save_original, render_options)


