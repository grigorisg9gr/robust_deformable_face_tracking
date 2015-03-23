import sys
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



def test_init_functions():
    assert(init_imports() == 1)   # confirm that all imports work

    p1 = './' + random_string_gen(25)
    msg = random_string_gen(23)
    from __init__ import check_if_path
    with captured_output() as (out, err):   # check_if_path prints the right message for non-existent paths
        check_if_path(p1, msg)
        output = out.getvalue().strip()
        assert(output == msg)

    # fake_path = '/tmp/test.fake/'
    # tt = check_if_path(fake_path, '')
    # mock1.assert_called_once_with(fake_path)


def test_pipeline_aux():
    from menpo.image import Image
    import numpy as np
    from menpo.shape import PointCloud
    test_img = Image(np.random.random([100, 100]))
    test_img.landmarks['PT'] = PointCloud([[20, 20], [20, 40], [40, 80], [40, 20]])
    from .pipeline_aux import *

    res_im = crop_rescale_img(test_img.copy())      # crop image and check reduced shapes
    assert(res_im.shape[0] < test_img.shape[0])
    assert(res_im.shape[1] < test_img.shape[1])

