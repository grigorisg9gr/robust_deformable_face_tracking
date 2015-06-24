import menpo.io as mio
from utils import (mkdir_p, print_fancy)
from utils.pipeline_aux import (read_public_images, check_img_type, check_path_and_landmarks, load_images,
                                check_initial_path, im_read_greyscale)
from utils.path_and_folder_definition import *  # import paths for databases, folders and libraries
from utils.clip import Clip
from joblib import Parallel, delayed

if __name__ == '__main__':
    args = len(sys.argv)
    path_0 = check_initial_path(args, sys.argv)

    if args > 3:
        in_landmarks_fol = str(sys.argv[2]) + '/'
        out_landmarks_fol = str(sys.argv[3]) + '/'
        print in_landmarks_fol, '   ', out_landmarks_fol
    else:
        in_landmarks_fol, out_landmarks_fol = '3_dlib_pred/', '4_fit_pbaam/'


from menpo.feature import no_op, fast_dsift
features = fast_dsift
patch_shape = (18, 18)  # (14,14)
crop_reading = 0.2  # 0.5
pix_thres = 250
diagonal_aam = 130

print_fancy('Building GN-DPMs for the clips')
path_clips = path_0 + frames
path_init_sh = path_0 + in_landmarks_fol
path_fitted_aam = path_0 + out_landmarks_fol

# read the images from the public databases (ibug, helen)
training_images = read_public_images(path_to_ibug, max_images=130, training_images=[], crop_reading=crop_reading, pix_thres=pix_thres) # 130
training_images = read_public_images(path_to_helen, max_images=320, training_images=training_images, crop_reading=crop_reading, pix_thres=pix_thres) #220
fitter = []


def process_frame(frame_name, clip):
    global fitter
    try:
        ln = mio.import_landmark_file(clip.path_read_ln + frame_name[:frame_name.rfind('.')] + '_0.pts')
    except:
        return
    im = im_read_greyscale(frame_name, clip.path_frames, img_type)
    if im is []:
        return
    im.landmarks['PTS2'] = ln
    fr = fitter.fit(im, im.landmarks['PTS2'].lms, gt_shape=None, crop_image=0.3)
    mio.export_landmark_file(fr.fitted_image.landmarks['final'], clip.path_write_ln + im.path.stem + '_0.pts', overwrite=True)


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
    if not check_path_and_landmarks(frames_path, clip_name, path_init_sh + clip_name):  # check that paths, landmarks exist
        return
    pts_folder = mkdir_p(path_fitted_aam + clip_name + '/')
    
    # loading images from the clip
    training_detector = load_images(list_frames, frames_path, path_init_sh, clip_name,
                                    training_images=list(training_images), max_images=110)  # make a new list of the concatenated images
    list_frames = sorted(list_frames)  # re-read frames, in case they are cut, re-arranged in loading
    
    print('\nBuilding Part based AAM for the clip ' + clip_name)
    aam = PartsAAMBuilder(parts_shape=patch_shape, features=features, diagonal=diagonal_aam,
                          normalize_parts=normalize_parts, scales=scales).build(training_detector, verbose=True)
    del training_detector
    
    fitter = PartsAAMFitter(aam, algorithm_cls=algorithm_cls, n_shape=n_shape,
                            n_appearance=n_appearance, sampling_mask=sampling_mask)
    del aam

    clip = Clip(clip_name, path_clips, frames, path_init_sh + clip_name + '/', pts_folder)
    Parallel(n_jobs=-1, verbose=4)(delayed(process_frame)(frame_name, clip) for frame_name in list_frames)
    fitter = []  # reset fitter



list_clips = sorted(os.listdir(path_clips))
img_type = check_img_type(list_clips, path_clips)  # assumption that all clips have the same extension, otherwise run in the loop for each clip separately.

[process_clip(clip_name) for clip_name in list_clips
 if not(clip_name in list_done)]


