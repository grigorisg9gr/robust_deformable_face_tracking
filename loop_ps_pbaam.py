
import matplotlib as mpl           # significant feature: For using the savefig in the python terminal. Should be added
mpl.use('Agg')                     # in the beginning of the program. http://stackoverflow.com/a/4935945/1716869
import menpo.io as mio
from utils import (mkdir_p, check_if_path)
from utils.pipeline_aux import (read_public_images, check_img_type, check_path_and_landmarks, load_images,
                                check_initial_path, im_read_greyscale)
from utils.path_and_folder_definition import *  # import paths for databases and folders from here
from joblib import Parallel, delayed
import shutil

if __name__ == '__main__':
    args = len(sys.argv)
    path_0 = check_initial_path(args, sys.argv)

    if args > 3:
        in_landmarks_fol = str(sys.argv[2]) + '/'
        in_landmarks_fol_test = str(sys.argv[3]) + '/'
        out_landmarks_fol = str(sys.argv[4]) + '/'
        print(in_landmarks_fol, '   ', out_landmarks_fol)
    else:
        in_landmarks_fol = '5_svm_faces/'
        in_landmarks_fol_test = '5_svm_faces/'
        out_landmarks_fol = '6_fit_pbaam/'


from menpo.feature import no_op, fast_dsift
features = fast_dsift
patch_shape = (18,18) #(14,14)
crop_reading = 0.2 # 0.5
pix_thres = 250

path_clips   = path_0 + frames 
path_init_sh = path_0 + in_landmarks_fol
path_read_sh = path_0 + in_landmarks_fol_test
path_fitted_aam = path_0 + out_landmarks_fol

# read the images from the public databases (ibug, helen)
training_images = read_public_images(path_to_ibug, max_images=130, training_images=[], crop_reading=crop_reading, pix_thres=pix_thres)
training_images = read_public_images(path_to_helen, max_images=400, training_images=training_images, crop_reading=crop_reading, pix_thres=pix_thres)
training_images = read_public_images(path_closed_eyes, max_images=100, training_images=training_images, crop_reading=crop_reading, pix_thres=pix_thres)

fitter = []


def process_frame(frame_name, clip_name, pts_folder, frames_path):
    global fitter
    name = frame_name[:frame_name.rfind('.')]
    try:
        ln = mio.import_landmark_file(path_init_sh + clip_name + '/' + name + '_0.pts')
        shutil.copy2(path_init_sh + clip_name + '/' + name + '_0.pts', pts_folder + name + '_0.pts')
        return      # if the landmark already exists, return (for performance improvement)
    except:
        pass
    try:
        ln = mio.import_landmark_file(path_read_sh + clip_name + '/' + name + '_0.pts')
    except:
        return
    im = im_read_greyscale(frame_name, frames_path, img_type)
    if im is []:
        return
    im.landmarks['PTS2'] = ln
    fr = fitter.fit(im, im.landmarks['PTS2'].lms, gt_shape=None, crop_image=0.3)
    res_im = fr.fitted_image
    mio.export_landmark_file(res_im.landmarks['final'], pts_folder + name + '_0.pts', overwrite=True)


from alabortijcv2015.aam import PartsAAMBuilder
from alabortijcv2015.aam import PartsAAMFitter
from alabortijcv2015.aam.algorithm import SIC


normalize_parts = False; scales = (1, .5)
algorithm_cls = SIC
sampling_step = 2
sampling_mask = np.require(np.zeros(patch_shape), dtype=np.bool)
sampling_mask[::sampling_step, ::sampling_step] = True
n_shape = [5, 13]; n_appearance = [50, 100]


def process_clip(clip_name):
    print('\nStarted processing of clip ' + clip_name)
    global fitter
    # paths and list of frames
    frames_path = path_clips + clip_name + '/'
    list_frames = sorted(os.listdir(frames_path))
    if not check_path_and_landmarks(frames_path, clip_name, path_init_sh + clip_name): # check that paths, landmarks exist
        return
    pts_folder = path_fitted_aam + clip_name + '/'; mkdir_p(pts_folder)
    
    # loading images from the clip
    training_detector = load_images(list_frames, frames_path, path_init_sh, clip_name,
                                    training_images=list(training_images), max_images=250)  # make a new list of the concatenated images
    list_frames = sorted(os.listdir(frames_path)) # re-read frames, in case they are cut, re-arranged in loading
    
    print('\nBuilding Part based AAM for the clip ' + clip_name)
    aam = PartsAAMBuilder(parts_shape=patch_shape, features=features, diagonal=180,
                          normalize_parts=normalize_parts, scales=scales).build(training_detector, verbose=True)
    del training_detector
    
    fitter = PartsAAMFitter(aam, algorithm_cls=algorithm_cls, n_shape=n_shape,
                            n_appearance=n_appearance, sampling_mask=sampling_mask)
    del aam

    Parallel(n_jobs=-1, verbose=4)(delayed(process_frame)(frame_name, clip_name, 
                                                          pts_folder, frames_path) for frame_name in list_frames)
    fitter =[] #reset fitter


list_clips = sorted(os.listdir(path_clips))
img_type = check_img_type(list_clips, path_clips)  # assumption that all clips have the same extension, otherwise run in the loop for each clip separately.
[process_clip(clip_name) for clip_name in list_clips 
 if not(clip_name in list_done)]


