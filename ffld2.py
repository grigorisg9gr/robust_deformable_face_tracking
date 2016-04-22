import menpo.io as mio
from utils import (mkdir_p, strip_separators_in_the_end, print_fancy, Logger)
from utils.path_and_folder_definition import *  # import paths for databases, folders and libraries
from utils.pipeline_aux import (check_img_type, im_read_greyscale, check_initial_path, check_path_and_landmarks, load_images)
from utils.clip import Clip
import numpy as np
from menpo.shape import PointCloud
from menpodetect.ffld2 import FFLD2Detector, train_ffld2_detector
from menpodetect.dlib.conversion import pointgraph_to_rect
from menpo.landmark import LandmarkGroup
from dlib import shape_predictor
from cyffld2 import load_model

predictor_dlib = shape_predictor(path_shape_pred)
detector = []


def main_for_ps_detector(path_clips, in_bb_fol, out_bb_fol, out_model_fol, out_landmarks_fol, overwrite=False):

    # define a dictionary for the paths
    paths = {}
    paths['clips'] = path_clips
    # existing bbox of detection
    paths['in_bb'] = path_clips + in_bb_fol
    # save bbox of detection
    paths['out_bb'] = path_clips + out_bb_fol
    paths['out_lns'] = path_clips + out_landmarks_fol
    # path that trained models will be saved.
    paths['out_model'] = mkdir_p(path_clips + out_model_fol)

    # Log file output.
    log = mkdir_p(path_clips + 'logs' + sep) + datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + \
          '_' + basename(__file__) + '.log'
    sys.stdout = Logger(log)

    print_fancy('Training person specific model with FFLD')
    list_clips = sorted(os.listdir(path_clips + frames))
    img_type = check_img_type(list_clips, path_clips + frames)
    negative_images = [i.as_greyscale(mode='channel', channel=1)
                       for i in mio.import_images(path_non_person_images,
                                                  normalise=False, max_images=300)]
    [process_clip(clip_name, paths, img_type, negative_images, overwrite=overwrite)
     for clip_name in list_clips];


def detection_to_pointgraph(detection):
    return PointCloud(np.array([(p.y, p.x) for p in detection.parts()]))

    
def predict_in_frame(frame_name, clip, img_type):
    global detector
    im = im_read_greyscale(frame_name, clip.path_frames, img_type, normalise=False)

    res_dlib = detector(im)
    # in the following lines a hack to figure out whether there are more than
    # 10 detections returned. In such a case, there should be two digits in
    # each group.
    num_res = len(res_dlib)
    if num_res == 0:
        return
    num1 = 1                # num1 and s1: Values if there are more than 10 detections in the image
    if num_res > 9:
        num1 = 2
    s1 = '%0' + str(num1)
    im_pili = np.array(im.as_PILImage())
    # loop over the returned detections.
    # By restricting the range below, we can choose the
    # k first (more confident) bounding boxes.
    # num_res to keep all, here keeping ONLY the most confident one
    for kk in range(0, 1):
        pts_end = im.path.stem + '_' + str(kk) + pts_type_out
        ln = im.landmarks['ffld2_' + (s1 + 'd') % kk]
        mio.export_landmark_file(ln, clip.path_write_ln[0] + pts_end, overwrite=True)
        # convert to landmarks
        det_frame = predictor_dlib(im_pili, pointgraph_to_rect(ln.lms))
        init_pc = detection_to_pointgraph(det_frame)
        mio.export_landmark_file(LandmarkGroup.init_with_all_label(init_pc),
                                 clip.path_write_ln[1] + pts_end, overwrite=True)


def process_clip(clip_name, paths, img_type, negative_images, overwrite=True):
    # overwrite: overwrite the training of the FFLD model.
    print(clip_name)
    frames_path = paths['clips'] + frames + clip_name + sep
    if not check_path_and_landmarks(frames_path, clip_name, paths['in_bb'] + clip_name + sep):
        return
    list_frames = sorted(os.listdir(frames_path))
    save_model = paths['out_model'] + clip_name + '.model'
    if (not os.path.exists(save_model)) or overwrite:
        # build the person specific detector. Firstly, load the images.
        training_pos = load_images(list_frames, frames_path, paths['in_bb'], clip_name, max_images=400)
        if len(training_pos) == 0:
            print('No positives found for the clip {}, skipping it.'.format(clip_name))
            return
        ps_model = train_ffld2_detector(training_pos, negative_images, n_components=1, n_relabel=6)
        ps_model.save(save_model)
    else:
        print('The model {} already exists and was loaded from disk.'.format(save_model))
        ps_model = load_model(save_model)
    global detector
    detector = FFLD2Detector(ps_model)

    p_det_bb = mkdir_p(paths['out_bb'] + clip_name + sep)
    p_det_landm = mkdir_p(paths['out_lns'] + clip_name + sep)
    clip = Clip(clip_name, paths['clips'], frames, write_ln=[p_det_bb, p_det_landm])
    # TODO: Try parallel model
    [predict_in_frame(frame_name, clip, img_type) for frame_name in list_frames]


if __name__ == '__main__':
    args = len(sys.argv)
    path_clips_m = check_initial_path(args, sys.argv)

    if 2 < args < 7:
        in_bb_fol_m = str(sys.argv[2]) + sep
        out_bb_fol_m = str(sys.argv[3]) + sep
        out_landmarks_fol_m = str(sys.argv[4]) + sep
        out_model_fol_m = strip_separators_in_the_end(out_bb_fol_m) + '_models' + sep
        print(in_bb_fol_m, '   ', out_landmarks_fol_m)
    else:
        in_bb_fol_m = '1_detect' + sep
        out_bb_fol_m, out_model_fol_m, out_landmarks_fol_m = '2_ffld' + sep, \
                                                             '2_ffld_models' + sep, '3_ffld_ln' + sep
    main_for_ps_detector(path_clips_m, in_bb_fol_m, out_bb_fol_m, out_model_fol_m, out_landmarks_fol_m, overwrite=False)


