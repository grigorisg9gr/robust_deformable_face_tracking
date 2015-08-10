import menpo.io as mio
from utils import (mkdir_p, print_fancy, Logger, strip_separators_in_the_end)
from utils.pipeline_aux import (read_public_images, check_img_type, check_path_and_landmarks, load_images,
                                check_initial_path, im_read_greyscale)
from utils.path_and_folder_definition import *  # import paths for databases, folders and libraries
from utils.clip import Clip
from menpo.io import export_pickle
from joblib import Parallel, delayed
# imports for GN-DPM builder/fitter:
from alabortijcv2015.aam import PartsAAMBuilder
from alabortijcv2015.aam import PartsAAMFitter
from alabortijcv2015.aam.algorithm import SIC


def main_for_ps_detector(path_clips, in_ln_fol, out_ln_fol, out_model_fol):
    # define a dictionary for the paths
    paths = {}
    paths['clips'] = path_clips
    paths['in_lns'] = path_clips + in_ln_fol  # existing bbox of detection
    paths['out_lns'] = path_clips + out_ln_fol
    paths['out_model'] = mkdir_p(path_clips + out_model_fol)  # path that trained models will be saved.

    # Log file output.
    log = mkdir_p(path_clips + 'logs' + sep) + datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + '_4_gndpm.log'
    sys.stdout = Logger(log)

    print_fancy('Building GN-DPMs for the clips')
    # read the images from the public databases (ibug, helen)
    training_images = read_public_images(path_to_ibug, max_images=130, training_images=[], crop_reading=crop_reading, pix_thres=pix_thres) # 130
    training_images = read_public_images(path_to_helen, max_images=220, training_images=training_images, crop_reading=crop_reading, pix_thres=pix_thres) #220
    training_images = read_public_images(path_closed_eyes, max_images=60, training_images=training_images, crop_reading=crop_reading, pix_thres=pix_thres)

    list_clips = sorted(os.listdir(path_clips + frames))
    img_type = check_img_type(list_clips, path_clips + frames)  # assumption that all clips have the same extension, otherwise run in the loop for each clip separately.
    t = [process_clip(clip_name, paths, in_ln_fol, training_images, img_type) for clip_name in list_clips
         if not(clip_name in list_done) and os.path.isdir(path_clips + frames + clip_name)]


from menpo.feature import no_op, fast_dsift
features = fast_dsift
patch_shape = (18, 18)  # (14,14)
crop_reading = 0.2  # 0.5
pix_thres = 250
diagonal_aam = 130
fitter = []
# gn-dpm params
normalize_parts = False; scales = (1, .5)
algorithm_cls = SIC
sampling_step = 2
sampling_mask = np.require(np.zeros(patch_shape), dtype=np.bool)
sampling_mask[::sampling_step, ::sampling_step] = True
n_shape = [3, 12]; n_appearance = [50, 100]


def process_frame(frame_name, clip, img_type):
    global fitter
    try:
        ln = mio.import_landmark_file(clip.path_read_ln + frame_name[:frame_name.rfind('.')] + '_0.pts')
    except:
        return
    im = im_read_greyscale(frame_name, clip.path_frames, img_type)
    if not im:
        return
    im.landmarks['PTS2'] = ln
    fr = fitter.fit(im, im.landmarks['PTS2'].lms, gt_shape=None, crop_image=0.3)
    mio.export_landmark_file(fr.fitted_image.landmarks['final'], clip.path_write_ln + im.path.stem + '_0.pts', overwrite=True)


def process_clip(clip_name, paths, in_ln_fol, training_images, img_type):
    global fitter
    # paths and list of frames
    frames_path = paths['clips'] + frames + clip_name + sep
    if not check_path_and_landmarks(frames_path, clip_name, paths['in_lns'] + clip_name):
        return False
    list_frames = sorted(os.listdir(frames_path))
    pts_folder = mkdir_p(paths['out_lns'] + clip_name + sep)
    
    # loading images from the clip
    training_detector = load_images(list(list_frames), frames_path, paths['in_lns'], clip_name,
                                    training_images=list(training_images), max_images=110)  # make a new list of the concatenated images
    
    print('\nBuilding Part based AAM for the clip {}.'.format(clip_name))
    aam = PartsAAMBuilder(parts_shape=patch_shape, features=features, diagonal=diagonal_aam,
                          normalize_parts=normalize_parts, scales=scales).build(training_detector, verbose=True)
    del training_detector
    
    fitter = PartsAAMFitter(aam, algorithm_cls=algorithm_cls, n_shape=n_shape,
                            n_appearance=n_appearance, sampling_mask=sampling_mask)
    # save the AAM model (requires plenty of disk space for each model).
    # aam.features = None
    # export_pickle(aam, paths['out_model'] + clip_name + '.pkl', overwrite=True)
    del aam

    clip = Clip(clip_name, paths['clips'], frames, in_ln_fol, pts_folder)
    Parallel(n_jobs=-1, verbose=4)(delayed(process_frame)(frame_name, clip, img_type) for frame_name in list_frames)
    fitter = []  # reset fitter
    return True


if __name__ == '__main__':
    args = len(sys.argv)
    path_0_m = check_initial_path(args, sys.argv)

    if args > 3:
        in_landmarks_fol_m = str(sys.argv[2]) + sep
        out_landmarks_fol_m = str(sys.argv[3]) + sep
        print(in_landmarks_fol_m, '   ', out_landmarks_fol_m)
    else:
        in_landmarks_fol_m, out_landmarks_fol_m = '3_dlib_pred' + sep, '4_fit_pbaam' + sep
    out_model_fol_m = strip_separators_in_the_end(out_landmarks_fol_m) + '_models' + sep
    main_for_ps_detector(path_0_m, in_landmarks_fol_m, out_landmarks_fol_m, out_model_fol_m)


