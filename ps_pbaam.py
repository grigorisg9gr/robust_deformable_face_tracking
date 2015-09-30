import menpo.io as mio
from utils import (mkdir_p, print_fancy, Logger, strip_separators_in_the_end)
from utils.pipeline_aux import (read_public_images, check_img_type, check_path_and_landmarks, load_images,
                                check_initial_path, im_read_greyscale)
from utils.path_and_folder_definition import *  # import paths for databases, folders and libraries
from utils.clip import Clip
from menpo.io import export_pickle
from joblib import Parallel, delayed
from shutil import copy2
# imports for GN-DPM builder/fitter:
from menpofit.aam import PatchAAM
from menpofit.aam.algorithm import WibergForwardCompositional as fit_alg
from menpofit.aam import LucasKanadeAAMFitter


def main_for_ps_aam(path_clips, in_ln_fol, out_ln_fol, out_model_fol, loop=False, mi=110, d_aam=130,
                    in_ln_fit_fol=None, max_helen=220, max_cl_e=60, n_shape=None, n_appearance=None):
    # loop: whether this is the 1st or the 2nd fit (loop).
    # define a dictionary for the paths
    assert(isinstance(n_shape, (int, float, type(None), list)))  # allowed values in menpofit.
    assert(isinstance(n_appearance, (int, float, type(None), list)))
    paths = {}
    paths['clips'] = path_clips
    paths['in_lns'] = path_clips + in_ln_fol  # existing bbox of detection
    paths['out_lns'] = path_clips + out_ln_fol
    paths['out_model'] = mkdir_p(path_clips + out_model_fol)  # path that trained models will be saved.
    if in_ln_fit_fol:
        paths['in_fit_lns'] = path_clips + in_ln_fit_fol
    else:
        paths['in_fit_lns'] = paths['in_lns']

    # Log file output.
    log = mkdir_p(path_clips + 'logs' + sep) + datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + '_4_gndpm.log'
    sys.stdout = Logger(log)

    print_fancy('Building GN-DPMs for the clips')
    # read the training images from the public databases (ibug, helen)
    tr_images = _aux_read_public_images(path_to_ibug, 130, [])
    tr_images = _aux_read_public_images(path_to_helen, max_helen, tr_images)
    tr_images = _aux_read_public_images(path_closed_eyes, max_cl_e, tr_images)

    list_clips = sorted(os.listdir(path_clips + frames))
    # assumption that all clips have the same extension, otherwise run in the loop for each clip separately:
    img_type = check_img_type(list_clips, path_clips + frames)
    t = [process_clip(clip_name, paths, tr_images, img_type, loop, mi=mi, d_aam=d_aam,
                      n_s=n_shape, n_a=n_appearance) for clip_name in list_clips
         if not(clip_name in list_done) and os.path.isdir(path_clips + frames + clip_name)]


from menpo.feature import fast_dsift
features = fast_dsift
patch_shape = (18, 18)
crop = 0.2  # crop when loading images from databases.
pix_thres = 250
fitter = []
# gn-dpm params
scales = (1, .5)
# alg_cls = fit_alg
sampling_step = 2
sampling_mask = np.require(np.zeros(patch_shape), dtype=np.bool)
sampling_mask[::sampling_step, ::sampling_step] = True
# n_shape = [3, 12]; n_appearance = [50, 100]
# n_shape = [5, 13]; n_appearance = [50, 100]


def process_frame(frame_name, clip, img_type, loop=False):
    # loop: whether this is part of GN-DPM second fit
    global fitter
    name = frame_name[:frame_name.rfind('.')]
    p0 = clip.path_read_ln[0] + name + '_0.pts'
    # find if this is 2nd fit or 1st.
    if loop:  # if 2nd fit, then if landmark is 'approved', return. Otherwise proceed.
        try:
            ln = mio.import_landmark_file(p0)
            copy2(p0, clip.path_write_ln + name + '_0.pts')
            return      # if the landmark already exists, return (for performance improvement)
        except ValueError:
            pass
        try:
            ln = mio.import_landmark_file(clip.path_read_ln[1] + name + '_0.pts')
        except ValueError:  # either not found or no suitable importer
            return
    else:
        try:
            ln = mio.import_landmark_file(p0)
        except ValueError:  # either not found or no suitable importer
            return
    im = im_read_greyscale(frame_name, clip.path_frames, img_type)
    if not im:
        return
    im.landmarks['PTS2'] = ln
    fr = fitter.fit_from_shape(im, im.landmarks['PTS2'].lms, crop_image=0.3)
    mio.export_landmark_file(fr.fitted_image.landmarks['final'], clip.path_write_ln + im.path.stem + '_0.pts', overwrite=True)


def process_clip(clip_name, paths, training_images, img_type, loop, mi=110, d_aam=130, n_s=None, n_a=None):
    # mi: max_images from clip, d_aam: diagonal of aam, n_s: n_shape, n_a: n_appearance
    global fitter
    # paths and list of frames
    frames_path = paths['clips'] + frames + clip_name + sep
    if not check_path_and_landmarks(frames_path, clip_name, paths['in_lns'] + clip_name):
        return False
    list_frames = sorted(os.listdir(frames_path))
    pts_folder = mkdir_p(paths['out_lns'] + clip_name + sep)
    
    # loading images from the clip
    training_detector = load_images(list(list_frames), frames_path, paths['in_lns'], clip_name,
                                    training_images=list(training_images), max_images=mi)
    
    print('\nBuilding Part based AAM for the clip {}.'.format(clip_name))
    aam = PatchAAM(training_detector, verbose=True, holistic_features=features, patch_shape=patch_shape,
                   diagonal=d_aam, scales=scales)
    del training_detector

    fitter = LucasKanadeAAMFitter(aam, lk_algorithm_cls=fit_alg, n_shape=n_s, n_appearance=n_a, sampling=sampling_mask)
    # save the AAM model (requires plenty of disk space for each model).
    aam.features = None
    export_pickle(aam, paths['out_model'] + clip_name + '.pkl', overwrite=True)
    aam.features = features
    del aam

    clip = Clip(clip_name, paths['clips'], frames, [paths['in_lns'], paths['in_fit_lns']], pts_folder)
    Parallel(n_jobs=-1, verbose=4)(delayed(process_frame)(frame_name, clip, img_type, loop) for frame_name in list_frames)
    fitter = []  # reset fitter
    return True


def _aux_read_public_images(path, max_images, training_images, crop=crop, pix_thres=pix_thres):
    return read_public_images(path, max_images=max_images, training_images=training_images,
                              crop_reading=crop, pix_thres=pix_thres)

if __name__ == '__main__':
    args = len(sys.argv)
    path_0_m = check_initial_path(args, sys.argv)

    if args > 3:
        in_landmarks_fol_m = str(sys.argv[2]) + sep
        out_landmarks_fol_m = str(sys.argv[3]) + sep
        print(in_landmarks_fol_m, '   ', out_landmarks_fol_m)
    else:
        in_landmarks_fol_m, out_landmarks_fol_m = '3_ffld_ln' + sep, '4_pbaam' + sep
    out_model_fol_m = strip_separators_in_the_end(out_landmarks_fol_m) + '_models' + sep
    main_for_ps_aam(path_0_m, in_landmarks_fol_m, out_landmarks_fol_m, out_model_fol_m,
                    n_shape=[3, 12], n_appearance=[50, 100])


