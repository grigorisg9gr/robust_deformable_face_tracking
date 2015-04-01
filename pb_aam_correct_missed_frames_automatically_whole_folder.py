
import matplotlib as mpl           # significant feature: For using the savefig in the python terminal. Should be added
mpl.use('Agg')                     # in the beginning of the program. http://stackoverflow.com/a/4935945/1716869
import menpo.io as mio
import os, sys


if __name__ == '__main__':
    args = len(sys.argv)
    if args > 1:
        path_0 = str(sys.argv[1])
        if not(os.path.isdir(path_0)):
            raise RuntimeError('The path %s does not exist as base folder' % path_0)
    else:
        raise RuntimeError('file not called with initial path')

    if args > 3:
        in_landmarks_fol = str(sys.argv[2]) + '/'
        out_landmarks_fol = str(sys.argv[3]) + '/'
        print in_landmarks_fol, '   ', out_landmarks_fol
    else:
        in_landmarks_fol = '3_dlib_pred/'
        out_landmarks_fol = '4_fit_pbaam/'



from utils import (mkdir_p, print_fancy)
from utils.pipeline_aux import (read_public_images, check_img_type, check_path_and_landmarks, load_images)
from utils.path_and_folder_definition import *  # import paths for databases, folders and visualisation options

from menpo.feature import no_op, fast_dsift
features = fast_dsift
patch_shape = (18,18) #(14,14)
crop_reading = 0.2 # 0.5
pix_thres = 250
diagonal_aam = 130

print_fancy('Building GN-DPMs for the clips')
path_clips   = path_0 + frames 
path_init_sh = path_0 + in_landmarks_fol
path_new_fit_vid = path_0 + foldvis + out_landmarks_fol; mkdir_p(path_new_fit_vid) #grigoris, check an ayta ta mkdir xreiazontai, afoy thewrhtika dhmioyrgei olo to path
path_fitted_aam = path_0 + out_landmarks_fol; mkdir_p(path_fitted_aam)

# read the images from the public databases (ibug, helen)
training_images = read_public_images(path_to_ibug, max_images=130, training_images=[], crop_reading=crop_reading, pix_thres=pix_thres) # 130
training_images = read_public_images(path_to_helen, max_images=320, training_images=training_images, crop_reading=crop_reading, pix_thres=pix_thres) #220

fitter = []


import matplotlib.pyplot as plt
import numpy as np
from random import randint
from joblib import Parallel, delayed



def process_frame(frame_name, clip_name, pts_folder, path_all_clip, frames_path): #, frames_path, path_all_clip, pts_folder, clip_name, fitter
    global fitter
    if frame_name[-4::]!=img_type: return # in case they are something different than an image
    if visual == 1: rand = randint(1,10000);  plt.figure(rand)
    # load image and a file of landmark points
#     print frame_name
    name = frame_name[:-4]; img_path = frames_path + name + img_type
    im = mio.import_image(img_path, normalise=True)
    try:
        ln = mio.import_landmark_file(path_init_sh + clip_name + '/' + name + '_0.pts')
    except:
        if visual == 1:
            viewer = im.view()
            viewer.save_figure(path_all_clip + name + img_type_out, pad_inches=0., overwrite=True, format=img_type_out[1::])
            # viewer.figure.savefig(path_all_clip + name + img_type_out)
            plt.close(rand)
        return
    im.landmarks['PTS2'] = ln # initial detector
    if im.n_channels == 3: im = im.as_greyscale(mode='luminosity')
    fr = fitter.fit(im, im.landmarks['PTS2'].lms, gt_shape=None, crop_image=0.3)
    res_im = fr.fitted_image

    mio.export_landmark_file(res_im.landmarks['final'], pts_folder + frame_name[:-4] + '_0.pts', overwrite=True)
    # plt.figure(rand) # ONLY need to create fig in every iteration, IF frames of the same video are plotted in parallel
    if visual == 1:
        res_im.crop_to_landmarks_proportion_inplace(0.3, group='final') # works, verified
        # viewer = res_im.view_landmarks(group='initial', render_numbering=False, lmark_view_kwargs={'colours': red}) #, avoid "new_figure=True" -> assigns new id
        # viewer = res_im.view_landmarks(group='final', render_numbering=False, lmark_view_kwargs={'colours': blue}) #, avoid "new_figure=True" -> assigns new id
        # viewer.figure.savefig(path_all_clip + name + img_type_out)
        viewer = res_im.view_landmarks(group='initial', render_numbering=False,  marker_face_colour=colour[0], marker_edge_colour=colour[0])
        viewer = res_im.view_landmarks(group='final', render_numbering=False,  marker_face_colour=colour[1], marker_edge_colour=colour[1])
        viewer.save_figure(path_all_clip + name + img_type_out, pad_inches=0., overwrite=True, format=img_type_out[1::])
        plt.close(rand) # plt.close('all') #problem with parallel # http://stackoverflow.com/a/21884375/1716869


from alabortijcv2015.aam import PartsAAMBuilder
from alabortijcv2015.aam import PartsAAMFitter
from alabortijcv2015.aam.algorithm import SIC


normalize_parts = False; scales = (1, .5)
algorithm_cls = SIC
sampling_step = 2
sampling_mask = np.require(np.zeros(patch_shape), dtype=np.bool)
sampling_mask[::sampling_step, ::sampling_step] = True
n_shape = [3, 12]; n_appearance = [50, 100]


def process_clip(clip_name):
    print('\nStarted processing of clip ' + clip_name)
    global fitter
    # paths and list of frames
    frames_path = path_clips + clip_name + '/'
    list_frames = sorted(os.listdir(frames_path))
    if not check_path_and_landmarks(frames_path, clip_name, path_init_sh + clip_name): # check that paths, landmarks exist
        return

    pts_folder = path_fitted_aam + clip_name + '/'; mkdir_p(pts_folder)
    path_all_clip = path_new_fit_vid + clip_name + '/'; mkdir_p(path_all_clip)

    
    # loading images from the clip
    training_detector = load_images(list_frames, frames_path, path_init_sh, clip_name,
                                    training_images=list(training_images), max_images=110)  # make a new list of the concatenated images
    list_frames = sorted(os.listdir(frames_path)) # re-read frames, in case they are cut, re-arranged in loading
    
    print('\nBuilding Part based AAM for the clip ' + clip_name)
    aam = PartsAAMBuilder(parts_shape=patch_shape, features=features, diagonal=diagonal_aam,
                            normalize_parts=normalize_parts, scales=scales).build(training_detector, verbose=True)
    n_channels = aam.appearance_models[0].mean().n_channels
    del training_detector
    
    fitter = PartsAAMFitter(aam, algorithm_cls=algorithm_cls, n_shape=n_shape,
                            n_appearance=n_appearance, sampling_mask=sampling_mask)
    del aam

#     [process_frame(frame_name, clip_name, pts_folder, path_all_clip, frames_path) for frame_name in list_frames];
    Parallel(n_jobs=-1, verbose=4)(delayed(process_frame)(frame_name, clip_name, 
                                                          pts_folder, path_all_clip, frames_path) for frame_name in list_frames)
    fitter =[] #reset fitter



list_clips = sorted(os.listdir(path_clips))
img_type = check_img_type(list_clips, path_clips)  # assumption that all clips have the same extension, otherwise run in the loop for each clip separately.

[process_clip(clip_name) for clip_name in list_clips
 if not(clip_name in list_done)]


