import menpo.io as mio
from utils import (mkdir_p, check_if_path)
from utils.path_and_folder_definition import *  # import paths for databases, folders and libraries
from utils.pipeline_aux import (check_img_type, im_read_greyscale, check_initial_path)
from utils.clip import Clip
import dlib
from menpodetect.dlib.conversion import pointgraph_to_rect
from menpo.shape import PointCloud
from joblib import Parallel, delayed
from menpo.landmark import LandmarkGroup

if __name__ == '__main__':
    args = len(sys.argv)
    path_clips = check_initial_path(args, sys.argv)

    if 2 < args < 5:
        in_bb_fol = str(sys.argv[2]) + '/'
        out_landmarks_fol = str(sys.argv[3]) + '/'
        print out_landmarks_fol, '   ', in_bb_fol
    else:
        in_bb_fol, out_landmarks_fol = '2_dpm/', '3_dlib_pred/'


# definition of paths
p_det_1 = path_clips + out_landmarks_fol
p_det_bb_0 = path_clips + in_bb_fol  # existing bbox of detection


def detection_to_pointgraph(detection):
    return PointCloud(np.array([(p.y, p.x) for p in detection.parts()]))

predictor_dlib = dlib.shape_predictor(path_shape_pred)


def predict_in_frame(frame_name, clip):
    im = im_read_greyscale(frame_name, clip.path_frames, img_type, normalise=False)
    if im is []:
        print frame_name, clip.path_frames
        return
    res = glob.glob(clip.path_read_ln + im.path.stem + '*.pts')
    # if len(res) == 0:
    #     return
    im_pili = np.array(im.as_PILImage())
    for kk, ln1 in enumerate(res):
        # load bbox
        ln = mio.import_landmark_file(ln1)
        im.landmarks['dlib_' + str(kk)] = ln
        # apply predictor
        pts_end = '_' + str(kk) + '.pts'
        init_pc = detection_to_pointgraph(predictor_dlib(im_pili,
                                                         pointgraph_to_rect(im.landmarks['dlib_' + str(kk)].lms)))
        mio.export_landmark_file(LandmarkGroup.init_with_all_label(init_pc),
                                 clip.path_write_ln + im.path.stem + pts_end, overwrite=True)
        # group_kk = 'bb_' + str(kk)
        # im.landmarks[group_kk] = init_pc
        # mio.export_landmark_file(im.landmarks[group_kk], p_det_landm + im.path.stem + pts_end, overwrite=True)


def process_clip(clip_name):
    p_det_bb = p_det_bb_0 + clip_name + '/'                      # load bbox of detection
    if not check_if_path(p_det_bb, 'Skipped clip ' + clip_name + ' because it does not have previous landmarks'):
        return
    print(clip_name)

    p_det_landm = p_det_1 + clip_name + '/'; mkdir_p(p_det_landm)
    clip = Clip(clip_name, path_clips, frames, p_det_bb, p_det_landm)
    list_frames = sorted(os.listdir(path_clips + frames + clip_name + '/'))
    Parallel(n_jobs=-1, verbose=4)(delayed(predict_in_frame)(frame_name, clip) for frame_name in list_frames);



# iterates over all clips in the folder and calls sequentially the function process_clip
list_clips = sorted(os.listdir(path_clips + frames))
img_type = check_img_type(list_clips, path_clips + frames)  # assumption that all clips have the same extension, otherwise run in the loop for each clip separately.
[process_clip(clip_name) for clip_name in list_clips
 if not(clip_name in list_done)]


