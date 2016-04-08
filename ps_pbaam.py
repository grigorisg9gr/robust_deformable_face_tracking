from utils import (mkdir_p, print_fancy, Logger, strip_separators_in_the_end)
from utils.pipeline_aux import (read_public_images, check_img_type, check_path_and_landmarks, load_images,
                                check_initial_path, im_read_greyscale)
from utils.path_and_folder_definition import *  # import paths for databases, folders and libraries
from utils.clip import Clip
from joblib import Parallel, delayed
from shutil import copy2
from menpo.io import export_pickle, import_landmark_file, export_landmark_file
from menpo.transform import PiecewiseAffine
from menpo.feature import fast_dsift
# imports for GN-DPM builder/fitter:
from menpofit.aam import PatchAAM
from menpofit.aam.algorithm import WibergForwardCompositional as fit_alg
from menpofit.aam import LucasKanadeAAMFitter

features = fast_dsift
patch_shape = (18, 18)
crop = 0.2  # crop when loading images from databases.
pix_thres = 250
fitter = []


def main_for_ps_aam(path_clips, in_ln_fol, out_ln_fol, out_model_fol, loop=False, mi=110, d_aam=130,
                    in_ln_fit_fol=None, max_helen=220, max_cl_e=60, n_shape=None, n_appearance=None,
                    out_ln_svm=None, patch_s_svm=(14, 14), pix_th_svm=170):
    """
    Main function for the person specific (part-based) AAM.
    Processes a batch of clips in the same folder. Creates the dictionary with the paths, the SVM params,
    loads the images from public datasets and then calls the processing per clip.
    :param path_clips:      str: Base path that contains the frames/lns folders.
    :param in_ln_fol:       str: Folder name for importing landmarks.
    :param out_ln_fol:      str: Folder name for exporting landmarks after AAM fit.
    :param out_model_fol:   str: Folder name for exporting the AAM (pickled file).
    :param loop:            bool: (optional) Declares whether this is a 2nd fit for AAM (loop).
    :param mi:              int: (optional) Max images of the clip loaded for the pbaam.
    :param d_aam:           int: (optional) Diagonal of AAM (param in building it).
    :param in_ln_fit_fol:   str: (optional) Folder name for importing during fitting (loop case).
    :param max_helen:       int: (optional) Max images of the helen dataset to be loaded.
    :param max_cl_e:        int: (optional) Max images of the 'close eyes' dataset to be loaded.
    :param n_shape:         int/list/None: (optional) Number of shapes for AAM (as expected in menpofit).
    :param n_appearance:    int/list/None: (optional) Number of appearances for AAM (as expected in menpofit).
    :param out_ln_svm:      str: (optional) Folder name for exporting landmarks after SVM (if applicable).
    :param patch_s_svm:     tuple: (optional) Patch size for SVM (if applicable).
    :param pix_th_svm:      int: (optional) Pixel threshold for resizing images in SVM classification.
    :return:
    """
    # loop: whether this is the 1st or the 2nd fit (loop).
    # define a dictionary for the paths
    assert(isinstance(n_shape, (int, float, type(None), list)))  # allowed values in menpofit.
    assert(isinstance(n_appearance, (int, float, type(None), list)))
    paths = {}
    paths['clips'] = path_clips
    paths['in_lns'] = path_clips + in_ln_fol  # existing bbox of detection
    paths['out_lns'] = path_clips + out_ln_fol
    paths['out_model'] = mkdir_p(path_clips + out_model_fol)  # path that trained models will be saved.
    paths['in_fit_lns'] = (path_clips + in_ln_fit_fol) if in_ln_fit_fol else paths['in_lns']
    paths['out_svm'] = (path_clips + out_ln_svm) if out_ln_svm else None

    # save the svm params in a dict in case they are required.
    svm_params = {}
    svm_params['apply'] = True if out_ln_svm else False  # True only if user provided path for output.
    # load pickled files for classifier and reference frame.
    if svm_params['apply']:
        print('Option to classify (non-)faces with SVM is activated.')
        svm_params['feat'] = features
        svm_params['patch_s'] = patch_s_svm
        name_p = features.__name__ + '_' + str(patch_s_svm[0]) + '_' + str(crop) + '_' + \
                 str(pix_th_svm) + '_' + 'helen_' + 'ibug_' + 'lfpw'
        path_pickle_svm = path_pickles + 'general_svm' + sep
        if not os.path.isdir(path_pickle_svm):
            raise RuntimeError('This path ({}) should contain the pickled file '
                               'for the SVM and the reference frame.'.format(path_pickle_svm))

        from sklearn.externals import joblib
        svm_params['clf'] = joblib.load(path_pickle_svm + name_p + '.pkl')
        svm_params['refFrame'] = joblib.load(path_pickle_svm + name_p + '_refFrame.pkl')

    # Log file output.
    log = mkdir_p(path_clips + 'logs' + sep) + datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + \
          '_' + basename(__file__) + '.log'
    sys.stdout = Logger(log)

    print_fancy('Building GN-DPMs for the clips')
    # read the training images from the public databases (ibug, helen)
    tr_images = _aux_read_public_images(path_to_ibug, 130, [])
    tr_images = _aux_read_public_images(path_to_helen, max_helen, tr_images)
    tr_images = _aux_read_public_images(path_closed_eyes, max_cl_e, tr_images)

    list_clips = sorted(os.listdir(path_clips + frames))
    # assumption that all clips have the same extension, otherwise run in the loop for each clip separately:
    img_type = check_img_type(list_clips, path_clips + frames)
    t = [process_clip(clip_name, paths, tr_images, img_type, loop, svm_params, mi=mi, d_aam=d_aam,
                      n_s=n_shape, n_a=n_appearance) for clip_name in list_clips
         if not(clip_name in list_done) and os.path.isdir(path_clips + frames + clip_name)]


def process_frame(frame_name, clip, img_type, svm_p, loop=False):
    """
    Applies the AAM fitter (global var) in a frame. Additionally, it might apply an
    SVM to verify it's a face if required.
    :param frame_name: str: Name of the frame along with extension, e.g. '000001.png'.
    :param clip:       str: Name of the clip.
    :param img_type:   str: Suffix (extension) of the frames, e.g. '.png'.
    :param svm_p:      dict: Required params for SVM classification.
    :param loop:       bool: (optional) Declares whether this is a 2nd fit for AAM (loop).
    :return:
    """
    global fitter
    name = frame_name[:frame_name.rfind('.')]
    p0 = clip.path_read_ln[0] + name + '_0.pts'
    # find if this is 2nd fit or 1st.
    if loop:  # if 2nd fit, then if landmark is 'approved', return. Otherwise proceed.
        try:
            ln = import_landmark_file(p0)
            copy2(p0, clip.path_write_ln[0] + name + '_0.pts')
            return      # if the landmark already exists, return (for performance improvement)
        except ValueError:
            pass
        try:
            ln = import_landmark_file(clip.path_read_ln[1] + name + '_0.pts')
        except ValueError:  # either not found or no suitable importer
            return
    else:
        try:
            ln = import_landmark_file(p0)
        except ValueError:  # either not found or no suitable importer
            return
    im = im_read_greyscale(frame_name, clip.path_frames, img_type)
    if not im:
        return
    im.landmarks['PTS2'] = ln
    fr = fitter.fit_from_shape(im, im.landmarks['PTS2'].lms, crop_image=0.3)
    p_wr = clip.path_write_ln[0] + im.path.stem + '_0.pts'
    export_landmark_file(fr.fitted_image.landmarks['final'], p_wr, overwrite=True)

    # apply SVM classifier by extracting patches (is face or not).
    if not svm_p['apply']:
        return
    im.landmarks.clear()  # temp solution
    im.landmarks['ps_pbaam'] = fr.fitted_image.landmarks['final']
    im_cp = im.crop_to_landmarks_proportion(0.2, group='ps_pbaam')
    im_cp = svm_p['feat'](im_cp)
    im2 = warp_image_to_reference_shape(im_cp, svm_p['refFrame'], 'ps_pbaam')
    _p_nd = im2.extract_patches_around_landmarks(group='source', as_single_array=True,
                                                 patch_shape=svm_p['patch_s']).flatten()
    if svm_p['clf'].decision_function(_p_nd) > 0:
        copy2(p_wr, clip.path_write_ln[1] + im.path.stem + '_0.pts')


def process_clip(clip_name, paths, training_images, img_type, loop, svm_params,
                 mi=110, d_aam=130, n_s=None, n_a=None):
    """
    Processes a clip. Accepts a clip (along with its params and paths), trains a person-specific
    part based AAM (pbaam) and then fits it to all the frames.
    :param clip_name:   str: Name of the clip.
    :param paths:       dict: Required paths for training/fitting/exporting data.
    :param training_images: list: List of menpo images (generic images) appended to the person specific ones.
    :param img_type:    str: Suffix (extension) of the frames, e.g. '.png'.
    :param loop:        bool: Declares whether this is a 2nd fit for AAM (loop).
    :param svm_params:  dict: Required params for SVM classification. If 'apply' is False,
    the rest are not used. Otherwise, requires reference frame and classifier loaded.
    :param mi:          int: (optional) Max images of the clip loaded for the pbaam.
    :param d_aam:       int: (optional) Diagonal of AAM (param in building it).
    :param n_s:         int/list/None: (optional) Number of shapes for AAM (as expected in menpofit).
    :param n_a:         int/list/None: (optional) Number of appearances for AAM (as expected in menpofit).
    :return:
    """
    global fitter
    # paths and list of frames
    frames_path = paths['clips'] + frames + clip_name + sep
    if not check_path_and_landmarks(frames_path, clip_name, paths['in_lns'] + clip_name):
        return False
    list_frames = sorted(os.listdir(frames_path))
    pts_p = mkdir_p(paths['out_lns'] + clip_name + sep)
    svm_p = mkdir_p(paths['out_svm'] + clip_name + sep)  # svm path
    
    # loading images from the clip
    training_detector = load_images(list(list_frames), frames_path, paths['in_lns'], clip_name,
                                    training_images=list(training_images), max_images=mi)
    
    print('\nBuilding Part based AAM for the clip {}.'.format(clip_name))
    aam = PatchAAM(training_detector, verbose=True, holistic_features=features, patch_shape=patch_shape,
                   diagonal=d_aam, scales=(.5, 1))
    del training_detector

    sampling_step = 2
    sampling_mask = np.zeros(patch_shape, dtype=np.bool)  # create the sampling mask
    sampling_mask[::sampling_step, ::sampling_step] = True
    fitter = LucasKanadeAAMFitter(aam, lk_algorithm_cls=fit_alg, n_shape=n_s, n_appearance=n_a, sampling=sampling_mask)
    # save the AAM model (requires plenty of disk space for each model).
    aam.features = None
    export_pickle(aam, paths['out_model'] + clip_name + '.pkl', overwrite=True)
    aam.features = features
    del aam

    clip = Clip(clip_name, paths['clips'], frames, [paths['in_lns'], paths['in_fit_lns']], [pts_p, svm_p])
    # [process_frame(frame_name, clip, img_type, svm_params,loop) for frame_name in list_frames];
    Parallel(n_jobs=-1, verbose=4)(delayed(process_frame)(frame_name, clip, img_type, svm_params,
                                                          loop) for frame_name in list_frames)
    fitter = []  # reset fitter
    return True


def warp_image_to_reference_shape(i, reference_frame, group):
    transform = [PiecewiseAffine(reference_frame.landmarks['source'][None], i.landmarks[group][None])]
    im2 = i.warp_to_mask(reference_frame.mask, transform[0])
    im2.landmarks['source'] = reference_frame.landmarks['source']
    return im2


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


