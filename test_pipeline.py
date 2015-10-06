from nose.tools import raises
from os.path import join, dirname, realpath, sep, isdir, getsize
from os import listdir
from sys import path
import tempfile


def random_string_gen(range1=12):
    import string
    import random
    return ''.join(random.choice(string.ascii_uppercase) for i in range(range1))

fake_path = join('tmp', 'file_fake', random_string_gen(15), '')
files_path = dirname(realpath(__file__)) + sep  # dir of the test files
path.append(files_path)  # hack for avoiding the ImportError ('no module named utils').


# ###############      AUX FUNCTIONS      ############### #


def aux_export_image_loop(base_p, frames, clip_name, iter1=10):
    # exports the same image a number of times (used for testing).
    from menpo.io import import_builtin_asset, export_image
    im = import_builtin_asset.lenna_png()
    from utils import mkdir_p
    p1 = mkdir_p(join(base_p, frames, clip_name, ''))
    for i in range(iter1):
        export_image(im, p1 + 'lenna_' + str(i) + '.png')


def aux_create_path_images():
    # creates a temp path, inserts some images there and returns the paths.
    # Used for testing functionality of the pipeline.
    from utils.path_and_folder_definition import frames
    clip = 'clip'
    # create temp path and the temp images
    tmp_base_path = join(tempfile.mkdtemp(suffix='_fake'), '')
    aux_export_image_loop(tmp_base_path, frames, clip)
    # run the dlib_predictor
    p_f = join(tmp_base_path, frames, clip, '')
    p_d = join(tmp_base_path, 'detect', clip, '')
    p_p = join(tmp_base_path, 'predict', clip, '')
    return tmp_base_path, p_f, p_d, p_p

# ####################################################### #

@raises(RuntimeError)
def test_run_pipeline_wrong_path():
    from run_pipeline import run_main
    run_main(fake_path)


def test_run_generic_detector():
    # run the dlib_predictor.py for some images and confirm that it has the expected output.
    from dlib_predictor import main_for_generic_detector

    tmp_base_path, p_f, p_d, p_p = aux_create_path_images()
    main_for_generic_detector(tmp_base_path, 'detect' + sep, 'predict' + sep)

    assert(isdir(p_d) and isdir(p_p))  # confirm that they are both valid paths
    assert(len(listdir(p_d)) == len(listdir(p_p)) == len(listdir(p_f)))

    # remove temp path
    from utils import rm_if_exists
    rm_if_exists(tmp_base_path)


def test_visualise_landmarks():
    from visualise_landmarks import main_call_visualisations
    from dlib_predictor import main_for_generic_detector
    from utils.path_and_folder_definition import foldvis

    tmp_base_path, p_f, p_d, p_p = aux_create_path_images()
    main_for_generic_detector(tmp_base_path, 'detect' + sep, 'predict' + sep)

    # 'visualise' method:
    main_call_visualisations(tmp_base_path, ['detect' + sep], 'visualise')
    p_v = join(tmp_base_path, foldvis, 'detect', 'clip', '')
    assert(isdir(p_v) and len(listdir(p_v)) == len(listdir(p_d)))
    assert(p_v + listdir(p_v)[0] > 10**5)  # the new visualised image is > 100 KB (not black visualisation).

    # remove temp path
    from utils import rm_if_exists
    rm_if_exists(tmp_base_path)
