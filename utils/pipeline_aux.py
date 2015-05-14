__author__ = 'gchrysos'
import os

# grigoris, files that are useful for the pipeline of detection.
# Several functions in this file require menpo or other menpo functionalities to execute

import menpo.io as mio
from menpo.feature import no_op
import glob
try:    ##### TODO: correct this hack with the proper way
    from robust_detection_and_localisation.utils import (check_if_path, find_image_type)
except:
    from utils import (check_if_path, find_image_type)

img_type = '.png'
def check_img_type(list_clips, path_clips, cnt=0):
    """
    Returns the extension of images in  a folder.
    It accepts a path and a list of clips, reads the clip number cnt, finds the
    images in that clip, reads the     first from the clip to define the extension.
    It uses the find_image_type(). If the list_clips is empty, it returns a default type.
    """
    global img_type
    if not check_if_path(path_clips):
        return img_type
    if len(list_clips) > cnt:
        p1 = path_clips + list_clips[cnt]
        if not check_if_path(p1):
            return img_type
        l1 = os.listdir(p1)
        if len(l1) == 0:
            raise RuntimeError('The path (%s) seems to be empty, so cannot find images or their extension' % p1)
        img_type = '.' + find_image_type(p1, l1[0])
        return img_type
    return img_type


def crop_rescale_img(im, crop_reading=0.3, pix_thres=230):
    """
    Auxiliary function that performs simple modification to an image:
    1) makes it greyscale,2) crops it around landmarks (group='PTS'), 3) rescales it
    """
    try:
        im.crop_to_landmarks_proportion_inplace(crop_reading)
    except ValueError:
        print('The image has %d groups of landmarks, could not perform cropping.' % im.landmarks.n_groups)
        return im
    if im.n_channels == 3:                                   # convert images to greyscale if they are not
        im = im.as_greyscale(mode='luminosity')
    if im.shape[0] > pix_thres or im.shape[1] > pix_thres:    # check that the images are not too big, otherwise resize them
        im = im.rescale_to_diagonal(pix_thres)
    return im    # necessary to return


def check_path_and_landmarks(frames_path, clip_name, landmark_path):
    """
    Checks for a valid path for frames and the existence of the respective landmark folder.
    If one of them does not exist, it returns False.
    It uses the check_if_path().
    """
    msg1 = 'Skipped clip ' + clip_name + ' because it is not a valid path'
    msg2 = 'Skipped clip ' + clip_name + ' because it does not have previous landmarks'
    return check_if_path(frames_path, msg1) and check_if_path(landmark_path, msg2)


def read_public_images(path_to_db, max_images=100, training_images=None, crop_reading=0.3, pix_thres=330, feat=None):
    """
    Read images from public databases. The landmarks are expected to be in the same folder.
    :param path_to_db:          Path to the folder of images. The landmark files are expected to be in the same folder.
    :param max_images:          Max images that will be loaded from this database. Menpo will try to load as many as requested (if they exist).
    :param training_images:     (optional) List of images to append the new ones.
    :param crop_reading:        (optional) Amount of cropping the image around the landmarks.
    :param pix_thres:           (optional) If the cropped image has a dimension bigger than this, it gets cropped to this diagonal dimension.
    :param feat:                (optional) Features to be applied to the images before inserting them to the list.
    :return:                    List of menpo images.
    """
    import menpo.io as mio
    if not(os.path.isdir(path_to_db)):
        raise RuntimeError('The path to the public DB images does not exist. Try with a valid path')
    if feat is None:
        feat = no_op
    if training_images is None:
        training_images = []
    for i in mio.import_images(path_to_db + '*', verbose=True, max_images=max_images):
        if not i.has_landmarks:
            continue
        i = crop_rescale_img(i, crop_reading=crop_reading, pix_thres=pix_thres)
        training_images.append(feat(i)) # append it to the list
    return training_images


def load_images(list_frames, frames_path, path_land, clip_name, max_images=None, training_images=None, crop_reading=0.3, pix_thres=330, feat=None):
    """
    Read images from the clips that are processed. The landmarks can be a different folder with the extension of pts and
    are searched as such.
    :param list_frames:         List of images that will be read and loaded.
    :param frames_path:         Path to the folder of images.
    :param path_land:           Path of the respective landmarks.
    :param clip_name:           The name of the clip being processed.
    :param max_images:          (optional) Max images that will be loaded from this clip.
    :param training_images:     (optional) List of images to append the new ones.
    :param crop_reading:        (optional) Amount of cropping the image around the landmarks.
    :param pix_thres:           (optional) If the cropped image has a dimension bigger than this, it gets cropped to this diagonal dimension.
    :param feat:                (optional) Features to be applied to the images before inserting them to the list.
    :return:                    List of menpo images.
    """
    from random import shuffle
    if not check_path_and_landmarks(frames_path, clip_name, path_land):
        return []
    if feat is None:
        feat = no_op
    if training_images is None:
        training_images = []
    #shuffle(list_frames)            # shuffle the list to ensure random ones are chosen
    if max_images is None:
        max_images = len(list_frames)
    elif max_images < 0:
        print('Warning: The images cannot be negative, loading the whole list instead.')
        max_images = len(list_frames)
    cnt = 0 # counter for images appended to the list
    for i, frame_name in enumerate(list_frames):
        name = frame_name[:-4]          # name of the image without the extension
        try:
            im = mio.import_image(frames_path + frame_name, normalise=True)
        except ValueError:                                      # in case the extension is unknown (by menpo)
            print('Ignoring the \'image\' %s' % frame_name)
            continue
        res = glob.glob(path_land + clip_name + '/' + name + '*.pts')
        if len(res) == 0:                       # if the image does not have any existing landmarks, ignore it
            continue
        elif len(res) > 1:
            #_r = randint(0,len(res)-1); #just for debugging reasons in different variable
            #ln = mio.import_landmark_file(res[_r]) # in case there are plenty of landmarks for the image, load random ones
            print('The image %s has more than one landmarks, for one person, loading only the first ones' % frame_name)
        ln = mio.import_landmark_file(res[0])
        im.landmarks['PTS'] = ln
        im = crop_rescale_img(im, crop_reading=crop_reading, pix_thres=pix_thres)
        training_images.append(feat(im))
        cnt += 1
        if cnt >= max_images:
            break # the limit of images (appended to the list) is reached
    return training_images

