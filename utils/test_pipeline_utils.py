import sys
from os.path import join, isdir, sep
from os import listdir
from mock import patch
from nose.tools import raises

fake_path = join('tmp', 'file_fake', '')


def random_string_gen(range1=12):
    import string
    import random
    return ''.join(random.choice(string.ascii_uppercase) for i in range(range1))

# redirect output for testing
from contextlib import contextmanager
try:                                # compatibility with python2 and python3.
    from StringIO import StringIO
except ImportError:
    from io import StringIO

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
        import errno
        import menpo
        import menpodetect
        import menpofit
        return 1
    except ImportError:
        return -1


def test_init_functions():
    assert(init_imports() == 1)   # confirm that all imports work

    p1 = './' + random_string_gen(25)
    msg = random_string_gen(23)
    from ..utils import check_if_path
    with captured_output() as (out, err):   # check_if_path prints the right message for non-existent paths.
        check_if_path(p1, msg)
        output = out.getvalue().strip()
        assert(output == msg)

    assert(check_if_path('.', '') is True)  # should return True, since this is a path.

    # fake_path = '/tmp/test.fake/'
    # tt = check_if_path(fake_path, '')
    # mock1.assert_called_once_with(fake_path)


def test_crop_rescale_img():
    from .pipeline_aux import crop_rescale_img
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


def test_pipeline_aux():
# @patch.object(pipeline_aux, 'load_images')
# def test_pipeline_aux(mock_load):
    from .pipeline_aux import check_img_type, load_images
    with patch('sys.stdout', new=StringIO()) as fake_out:                          # check_img_type for non valid path
        check_img_type(['greg', 'l1', 'm1'], fake_path)
        assert('valid path' in fake_out.getvalue())


def test_pipeline_aux_load_images():
    from .pipeline_aux import load_images
    with patch('sys.stdout', new=StringIO()) as fake_out:
        # check for negative number of images as max_images
        load_images(listdir('.'), '.', '.', '', max_images=-5)
        assert('negative' in fake_out.getvalue())
        # since python files (*.py) are here, it will fail reading an image at least once.
        assert('Ignoring' in fake_out.getvalue())

    with patch('sys.stdout', new=StringIO()) as fake_out:         # check for positive number of images
        load_images(listdir('.'), '.', '.', '', max_images=5)
        assert('negative' not in fake_out.getvalue())

    with patch('sys.stdout', new=StringIO()) as fake_out:         # check for not valid path
        if not isdir(fake_path):
            ret = load_images(listdir('.'), fake_path, fake_path, '', max_images=5)
            print(fake_out)
            assert('not a valid path' in fake_out.getvalue())
            assert(ret == [])


@raises(AssertionError)
def test_strip_separators_in_the_end_error():
    from ..utils import strip_separators_in_the_end
    strip_separators_in_the_end(9)


def test_strip_separators_in_the_end():
    from ..utils import strip_separators_in_the_end
    name = 'fake1'
    name1 = name + sep
    assert(name == strip_separators_in_the_end(name1))  # one sep in the end
    assert(name == strip_separators_in_the_end(name))  # no sep in the end

    name1 += sep*3
    assert(name == strip_separators_in_the_end(name1))  # several sep in the end

    assert('' == strip_separators_in_the_end(''))  # several sep in the end
