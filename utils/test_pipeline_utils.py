import sys, os
from mock import patch


def random_string_gen(range1=12):
    import string
    import random
    return ''.join(random.choice(string.ascii_uppercase) for i in range(range1))

# redirect output for testing
from contextlib import contextmanager
from StringIO import StringIO

@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def init_imports():
    try:
        import os, shutil, errno
        import menpo, menpodetect, menpofit
        return 1
    except:
        return -1

fake_path = '/tmp/test.fake/'

def test_init_functions():
    assert(init_imports() == 1)   # confirm that all imports work

    p1 = './' + random_string_gen(25)
    msg = random_string_gen(23)
    from __init__ import check_if_path
    with captured_output() as (out, err):   # check_if_path prints the right message for non-existent paths
        check_if_path(p1, msg)
        output = out.getvalue().strip()
        assert(output == msg)

    assert(check_if_path('.', '') is True) # should return True, since this is a path

    # fake_path = '/tmp/test.fake/'
    # tt = check_if_path(fake_path, '')
    # mock1.assert_called_once_with(fake_path)

from .pipeline_aux import *
# @patch.object(pipeline_aux, 'load_images')
def test_pipeline_aux():
# def test_pipeline_aux(mock_load):
    from menpo.image import Image
    import numpy as np
    from menpo.shape import PointCloud
    test_img = Image(np.random.random([100, 100]))
    test_img.landmarks['PT'] = PointCloud([[20, 20], [20, 40], [40, 80], [40, 20]])

    res_im = crop_rescale_img(test_img.copy())      # crop image and check reduced shapes
    assert(res_im.shape[0] < test_img.shape[0])
    assert(res_im.shape[1] < test_img.shape[1])

    res_im2 = crop_rescale_img(test_img.copy(), crop_reading=1, pix_thres=40)      # check pixel threshold
    assert(res_im2.shape[0] < test_img.shape[0])

    res_im3 = crop_rescale_img(test_img.copy(), crop_reading=1, pix_thres=400)     # test that the image remains the same
    assert(res_im3.shape[1] > test_img.shape[1]-3)

    with patch('sys.stdout', new=StringIO()) as fake_out:                          # check_img_type for non valid path
        check_img_type(['greg', 'l1', 'm1'], fake_path)
        assert('valid path' in fake_out.getvalue())

    # print os.listdir('.')
    # with captured_output() as (out, err):
    #     load_images(os.listdir('.'), fake_path, fake_path, '', max_images=-5) # check for negative images
    #     output = out.getvalue().strip()
    #     assert('negative' in output)
    # print output



def test_pipeline_aux_load_images():
    with patch('sys.stdout', new=StringIO()) as fake_out:
        load_images(os.listdir('.'), '.', '.', '', max_images=-5) # check for negative number of images
        assert('negative' in fake_out.getvalue())
        assert('Ignoring' in fake_out.getvalue())                 # since python files are here, at least once it should fail to read an image

    with patch('sys.stdout', new=StringIO()) as fake_out:         # check for positive number of images
        load_images(os.listdir('.'), '.', '.', '', max_images=5)
        assert('negative' not in fake_out.getvalue())

    with patch('sys.stdout', new=StringIO()) as fake_out:         # check for not valid path
        if not os.path.isdir(fake_path):
            ret = load_images(os.listdir('.'), fake_path, fake_path, '', max_images=5)
            print fake_out
            assert('not a valid path' in fake_out.getvalue())
            assert(ret == [])




