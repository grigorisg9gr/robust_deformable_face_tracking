
import menpo.io as mio
from utils import (mkdir_p, check_if_path)
from utils.path_and_folder_definition import *  # import paths for databases, folders and libraries
from utils.pipeline_aux import (check_img_type, im_read_greyscale, check_initial_path)
from utils.clip import Clip
import dlib
from menpodetect.dlib.conversion import pointgraph_to_rect
from menpodetect import load_dlib_frontal_face_detector
from menpo.shape import PointCloud
from menpo.landmark import LandmarkGroup
from joblib import Parallel, delayed

if __name__ == '__main__':
    args = len(sys.argv)
    path_clips = check_initial_path(args, sys.argv)

    if 2 < args < 5:
        out_landmarks_fol = str(sys.argv[3]) + '/'
        out_bb_fol = str(sys.argv[2]) + '/'
        print out_landmarks_fol, '   ', out_bb_fol
    else:
        out_landmarks_fol, out_bb_fol = '1_dlib_pred/', '1_dlib_detect/'


# definition of paths
p_det_1 = path_clips + out_landmarks_fol
p_det_bb_0 = path_clips + out_bb_fol       # save bbox of detection


def detection_to_pointgraph(detection):
    return PointCloud(np.array([(p.y, p.x) for p in detection.parts()]))

dlib_init_detector = load_dlib_frontal_face_detector()
predictor_dlib = dlib.shape_predictor(path_shape_pred)


def detect_in_frame(frame_name, clip):
    im = im_read_greyscale(frame_name, clip.path_frames, img_type, normalise=False)
    if im is []:
        print frame_name, clip.path_frames
        return
    # call dlib detector
    res_dlib = dlib_init_detector(im, group_prefix='dlib')
    num_res = len(res_dlib)
    if num_res == 0:
        return
    num1 = 1                # num1 and s1: Values if there are more than 10 detections in the image
    if num_res > 9:
        num1 = 2
    s1 = '%0' + str(num1)
    im_pili = im.as_PILImage()
    for kk in range(num_res):
        group_bb = 'dlib_' + (s1 + 'd') % kk
        pts_end = im.path.stem + '_' + str(kk) + pts_type_out  # define the ending of each pts that will be exported
        mio.export_landmark_file(im.landmarks[group_bb], clip.path_write_ln[0] + pts_end, overwrite=True)
        # from bounding box to points (dlib predictor)
        init_pc = detection_to_pointgraph(predictor_dlib(np.array(im_pili), pointgraph_to_rect(im.landmarks[group_bb].lms)))
        mio.export_landmark_file(LandmarkGroup.init_with_all_label(init_pc), clip.path_write_ln[1] + pts_end, overwrite=True)



def process_clip(clip_name):
    """
    Function that processes one clip. It creates the essential paths (for landmarks and visualisations)
    and then calls the function detect_in_frame for each frame of the clip.
    """
    frames_path = path_clips + frames + clip_name + '/'
    list_frames = sorted(os.listdir(frames_path))
    if not check_if_path(frames_path, 'Skipped clip ' + clip_name + ' because its path of frames is not valid'):
        return
    print(clip_name)
    p_det_bb = p_det_bb_0 + clip_name + '/'; mkdir_p(p_det_bb)  # save bbox of detection
    p_det_landm = p_det_1 + clip_name + '/'; mkdir_p(p_det_landm)
    clip = Clip(clip_name, path_clips, frames, write_ln=[p_det_bb, p_det_landm])

    # Parallel(n_jobs=-1, verbose=4)(delayed(detect_in_frame)(frame_name, clip) for frame_name in list_frames);
    t = [detect_in_frame(frame_name, clip) for frame_name in list_frames]


# iterates over all clips in the folder and calls sequentially the function process_clip
list_clips = sorted(os.listdir(path_clips + frames))
img_type = check_img_type(list_clips, path_clips + frames)  # assumption that all clips have the same extension, otherwise run in the loop for each clip separately.
[process_clip(clip_name) for clip_name in list_clips
 if not(clip_name in list_done)]

