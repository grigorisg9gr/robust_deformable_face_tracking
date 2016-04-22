
from utils import mkdir_p, check_if_path, Logger
from utils.path_and_folder_definition import *  # import paths for databases, folders and libraries
from utils.pipeline_aux import (check_img_type, im_read_greyscale, check_initial_path)
from utils.clip import Clip
from dlib import shape_predictor
from menpo.io import export_landmark_file
from menpo.shape import PointCloud
from menpodetect.dlib.conversion import pointgraph_to_rect
from menpodetect import load_dlib_frontal_face_detector
from joblib import Parallel, delayed

dlib_init_detector = load_dlib_frontal_face_detector()
predictor_dlib = shape_predictor(path_shape_pred)
# define a lambda function that accepts the image, along with the bounding box
# and returns the landmark localisation outcome.
f = lambda imp, ln_g: detection_to_pointgraph(predictor_dlib(imp, pointgraph_to_rect(ln_g)))


def main_for_generic_detector(path_clips, out_bb_fol, out_landmarks_fol):
    """
    Main function for the generic detection step.
    Processes a batch of clips in the same folder. Creates the dictionary with the paths, calls
    the processing per clip.
    :param path_clips:      str: Base path that contains the frames/lns folders.
    :param out_bb_fol:      str: Folder name for exporting the bounding box of the detection.
    :param out_landmarks_fol: str: Folder name for exporting the landmarks of the predictor.
    :return:
    """
    # define a dictionary for the paths
    paths = {}
    paths['clips'] = path_clips
    paths['out_bb'] = path_clips + out_bb_fol       # path for bbox of detection
    paths['out_lns'] = path_clips + out_landmarks_fol

    # Log file output.
    log = mkdir_p(path_clips + 'logs' + sep) + datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + \
          '_' + basename(__file__) + '.log'
    sys.stdout = Logger(log)

    # iterates over all clips in the folder and calls sequentially the function process_clip
    list_clips = sorted(os.listdir(path_clips + frames))
    img_type = check_img_type(list_clips, path_clips + frames)
    [process_clip(clip_name, paths, img_type) for clip_name in list_clips
     if not(clip_name in list_done)]


def detection_to_pointgraph(detection):
    return PointCloud(np.array([(p.y, p.x) for p in detection.parts()]))


def detect_in_frame(frame_name, clip, img_type):
    # if normalise=True in im_read_greyscale: before calling dlib detector, image should be converted to uint8
    im = im_read_greyscale(frame_name, clip.path_frames, img_type, normalise=False)
    if not im:
        print(frame_name, clip.path_frames)
        return
    res_dlib = dlib_init_detector(im, group_prefix='dlib')
    im_pili = np.array(im.as_PILImage())
    for kk, g in enumerate(im.landmarks.group_labels):
        # define the ending of each pts that will be exported
        pts_end = im.path.stem + '_' + str(kk) + pts_type_out
        export_landmark_file(im.landmarks[g], clip.path_write_ln[0] + pts_end, overwrite=True)
        # from bounding box to points (dlib predictor)
        # even though the landmark points are not strictly required in this step, since in
        # the next step, we are going to use the bounding boxes for the person, specific
        # detection, you can export them (or not), by (un-)commenting the three lines below.
        im.landmarks['_tmp'] = f(im_pili, im.landmarks[g].lms)
        export_landmark_file(im.landmarks['_tmp'], clip.path_write_ln[1] + pts_end, overwrite=True)


def process_clip(clip_name, paths, img_type):
    """
    It processes one clip. It creates the essential paths (for the bounding
    box and the respective landmarks) and then calls the function
    detect_in_frame for each frame of the clip.
    :param clip_name:       str: Name of the clip.
    :param paths:           dict: Required paths for training/fitting/exporting data.
        'clips':            str: Parent folder of the clips.  All clips
    should be in sub-folders in the (path_clips + frames) folder.
        'out_bb':           str: Parent path that the bounding boxes will be saved.
        'out_lns':          str: Parent path that the landmarks will be saved.
    :param img_type:        str: Suffix (extension) of the frames, e.g. '.png'.
    :return:
    """
    frames_path = paths['clips'] + frames + clip_name + sep
    list_frames = sorted(os.listdir(frames_path))
    if not isdir(frames_path):
        ms1 = 'The frames folder {} does not seem to exist. Skipping the clip.'
        print(ms1.format(frames_path))
        return
    print(clip_name)
    # define the output paths and make their directories.
    p_det_bb = mkdir_p(paths['out_bb'] + clip_name + sep)
    p_det_landm = mkdir_p(paths['out_lns'] + clip_name + sep)
    clip = Clip(clip_name, paths['clips'], frames, write_ln=[p_det_bb, p_det_landm])

    Parallel(n_jobs=-1, verbose=4)(delayed(detect_in_frame)(frame_name, clip, img_type)
                                   for frame_name in list_frames);


if __name__ == '__main__':
    # in case it is called directly through the terminal.
    # expected input:
    #   1) path base (frames, and output landmarks will be written there).
    #   2) (optional) output name for bounding box folders.
    #   3) (optional) output name for landmark files folders.
    args = len(sys.argv)
    path_clips_m = check_initial_path(args, sys.argv)

    if 2 < args < 5:
        out_landmarks_fol_m = str(sys.argv[3]) + sep
        out_bb_fol_m = str(sys.argv[2]) + sep
        print(out_landmarks_fol_m, '   ', out_bb_fol_m)
    else:
        out_landmarks_fol_m, out_bb_fol_m = '1_pred' + sep, '1_detect' + sep
    main_for_generic_detector(path_clips_m, out_bb_fol_m, out_landmarks_fol_m)

