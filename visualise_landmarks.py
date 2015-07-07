import matplotlib as mpl           # significant feature: For using the savefig in the python terminal. Should be added
mpl.use('Agg')
from utils import mkdir_p
from utils.pipeline_aux import (check_path_and_landmarks, check_img_type, check_initial_path)
from utils.visualisation_aux import generate_frames_max_bbox
from utils.path_and_folder_definition import *  # import paths for databases, folders and visualisation options
import re

if __name__ == '__main__':
    args = len(sys.argv)
    path_clips = check_initial_path(args, sys.argv)

    list_landm = []
    if args > 2:  # all translated as paths of landmarks to be saved
        if sys.argv[2] == 'c':
            method = 'compare'
        elif sys.argv[2] == 'h':
            print('Call like this: python visualise_landmarks.py c [path] [ln1 ln2 ...] for comparison mode.')
            print('Call like this: python visualise_landmarks.py m [path] [ln1 ln2 ...] for simple visualisation mode.')
            exit(1)
        else:
            method = 'visualise'
        for i in range(3, args):
            list_landm.append(sys.argv[i])
    else:
        method = 'visualise'
        list_landm = ['1_dlib_pred']


# define 'constant' values and viewing options
proportion = 0.2
figure_size = (10, 8)
overwrite = True
save_original = True

# paths and format of images
pts_format = ['_0' + pts_type_out]
path_f = path_clips + frames
list_clips = sorted(os.listdir(path_f))
frames_format = check_img_type(list_clips, path_f)

def for_each_landmark_folder(fold_landm, save_path_0):
    save_path = mkdir_p(save_path_0 + fold_landm + '/')
    for clip in list_clips:
        path_frames = path_f + clip + '/'
        path_lndm = path_clips + fold_landm + '/' + clip + '/'
        if not check_path_and_landmarks(path_frames, clip, path_lndm):
            continue
        print(clip)
        save_path_i = mkdir_p(save_path + clip + '/')
        imgs = generate_frames_max_bbox(path_frames, frames_format, [path_lndm],
                                        pts_format, [fold_landm],
                                        save_path_i, proportion, figure_size, overwrite, save_original, render_options)


def simple_visualise(path_clips, list_landm):
    print('Simple visualisation of individual landmark techniques was chosen.')
    save_path_0 = path_clips + foldvis + '/'
    for landm in list_landm:
        if not check_if_path(path_clips + landm + '/', ''):
            continue
        for_each_landmark_folder(landm, save_path_0)


def ln_folder_existence(list_landm, path_clips):
    list_landm_f = []
    for landm in list_landm:
        if not check_if_path(path_clips + landm + '/', ''):
            continue
        list_landm_f.append(landm)
    if len(list_landm_f) == 0:
        print('No valid landmark folders found from the list.')
        exit(1)
    return list_landm_f


def output_folder_name(list_landm_f, save_path_1):
    # define the name of the output folder
    _pattern = re.compile('[^a-zA-Z0-9]+')
    name_fold = ''.join(map(lambda x: '%s_' % _pattern.sub('', x), list_landm_f))
    name_fold = name_fold[:-1]          # stripping the last _
    save_path_2 = mkdir_p(save_path_1 + name_fold + '/')
    return save_path_2


def compare_main(path_clips, list_landm, pts_format):
    # main method for comparing landmarks. Old compare_landmarks.py option.
    print('Comparison method was chosen.')
    save_path_1 = path_clips + foldvis + '/' + foldcmp + '/'
    list_landm_f = ln_folder_existence(list_landm, path_clips)
    pts_format *= len(list_landm_f)     # replicate list elements as many times as the different landmark groups
    save_path_2 = output_folder_name(list_landm_f, save_path_1)

    for clip in list_clips:
        path_frames = path_f + clip + '/'
        path_landm = []
        for landm in list_landm_f:
            if not check_if_path(path_clips + landm + '/' + clip + '/', ''):
                continue
            path_landm.append(path_clips + landm + '/' + clip + '/')
        print(clip)
        save_path_i = mkdir_p(save_path_2 + clip + '/')
        imgs = generate_frames_max_bbox(path_frames, frames_format, path_landm,
                                        pts_format, list_landm_f,
                                        save_path_i, proportion, figure_size, overwrite, save_original, render_options)


if method == 'compare':
    compare_main(path_clips, list_landm, pts_format)
else:
    simple_visualise(path_clips, list_landm)