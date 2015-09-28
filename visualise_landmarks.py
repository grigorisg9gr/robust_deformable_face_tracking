import matplotlib as mpl           # significant feature: For using the savefig in the python terminal. Should be added
mpl.use('Agg')
from utils import mkdir_p
from utils.pipeline_aux import (check_path_and_landmarks, check_img_type, check_initial_path)
from utils.visualisation_aux import generate_frames_max_bbox
from utils.path_and_folder_definition import *  # import paths for databases, folders and visualisation options
import re


def main_call_visualisations(path_clips, list_landm, method):
    # define 'constant' values and viewing options
    viewing_options = {}
    viewing_options['proportion'] = 0.2
    viewing_options['figure_size'] = (10, 8)
    viewing_options['overwrite'] = True
    viewing_options['save_original'] = False

    # paths and format of images
    path_related = {}
    path_related['path_clips'] = path_clips
    path_related['list_landm'] = list_landm
    path_related['pts_format'] = ['_0' + pts_type_out]
    path_related['path_f'] = path_clips + frames
    path_related['list_clips'] = sorted(os.listdir(path_related['path_f']))
    path_related['frames_format'] = check_img_type(path_related['list_clips'], path_related['path_f'])

    if method == 'compare':
        compare_main(path_related, viewing_options)
    elif method == 'black_visualise':  # visualise landmarks only
        simple_visualise(path_related, viewing_options, only_ln=True)
    else:
        simple_visualise(path_related, viewing_options)


def for_each_landmark_folder(fold_landm, save_path_0, path_related, viewing_options, only_ln=False):
    save_path = mkdir_p(save_path_0 + fold_landm + sep)
    for clip in path_related['list_clips']:
        path_frames = path_related['path_f'] + clip + sep
        path_lndm = path_related['path_clips'] + fold_landm + sep + clip + sep
        if not check_path_and_landmarks(path_frames, clip, path_lndm):
            continue
        print(clip)
        save_path_i = mkdir_p(save_path + clip + sep)
        try:  # hack, problem with boundaries
            imgs = generate_frames_max_bbox(path_frames, path_related['frames_format'], [path_lndm],
                                        path_related['pts_format'], [fold_landm], save_path_i,
                                        viewing_options['proportion'], viewing_options['figure_size'],
                                        viewing_options['overwrite'], viewing_options['save_original'],
                                        render_options, only_ln=only_ln)
        except:
            pass


def simple_visualise(path_related, viewing_options, only_ln=False):
    print('Simple visualisation of individual landmark techniques was chosen.')
    save_path_0 = path_related['path_clips'] + foldvis + sep
    for landm in path_related['list_landm']:
        if not check_if_path(path_related['path_clips'] + landm + sep, ''):
            continue
        for_each_landmark_folder(landm, save_path_0, path_related, viewing_options, only_ln=only_ln)


def ln_folder_existence(list_landm, path_clips):
    list_landm_f = []
    for landm in list_landm:
        if not check_if_path(path_clips + landm + sep, ''):
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
    save_path_2 = mkdir_p(save_path_1 + name_fold + sep)
    return save_path_2


def compare_main(path_related, viewing_options):
    # main method for comparing landmarks. Old compare_landmarks.py option.
    print('Comparison method was chosen.')
    save_path_1 = path_related['path_clips'] + foldvis + sep + foldcmp + sep
    list_landm_f = ln_folder_existence(path_related['list_landm'], path_related['path_clips'])
    path_related['pts_format'] *= len(list_landm_f)     # replicate list elements as many times as the different landmark groups
    save_path_2 = output_folder_name(list_landm_f, save_path_1)

    for clip in path_related['list_clips']:
        path_frames = path_related['path_f'] + clip + sep
        path_landm = []
        for landm in list_landm_f:
            if not check_if_path(path_related['path_clips'] + landm + sep + clip + sep, ''):
                continue
            path_landm.append(path_related['path_clips'] + landm + sep + clip + sep)
        print(clip)
        save_path_i = mkdir_p(save_path_2 + clip + sep)
        imgs = generate_frames_max_bbox(path_frames, path_related['frames_format'], path_landm,
                                        path_related['pts_format'], list_landm_f,
                                        save_path_i, viewing_options['proportion'],
                                        viewing_options['figure_size'], viewing_options['overwrite'],
                                        viewing_options['save_original'], render_options)


if __name__ == '__main__':
    args = len(sys.argv)
    path_clips_m = check_initial_path(args, sys.argv)

    list_landm_m = []
    if args > 2:  # all translated as paths of landmarks to be saved
        if sys.argv[2] == 'c':
            method_m = 'compare'
        elif sys.argv[2] == 'h':
            print('Call like this: python visualise_landmarks.py c [path] [ln1 ln2 ...] for comparison mode.')
            print('Call like this: python visualise_landmarks.py m [path] [ln1 ln2 ...] for simple visualisation mode.')
            exit(1)
        elif sys.argv[2] == 'vln':
            method_m = 'black_visualise'
        else:
            method_m = 'visualise'
        for i in range(3, args):
            list_landm_m.append(sys.argv[i])
    else:
        method_m = 'visualise'
        list_landm_m = ['1_dlib_pred']
    main_call_visualisations(path_clips_m, list_landm_m, method_m)
