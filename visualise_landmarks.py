import matplotlib as mpl           # significant feature: For using the savefig in the python terminal. Should be added
mpl.use('Agg')
from utils import (mkdir_p, check_if_path)
from utils.pipeline_aux import (check_path_and_landmarks, check_img_type, check_initial_path)
from utils.visualisation_aux import generate_frames_max_bbox
from utils.path_and_folder_definition import *  # import paths for databases, folders and visualisation options

if __name__ == '__main__':
    args = len(sys.argv)
    path_clips = check_initial_path(args, sys.argv)

    list_landm = []
    if args > 2: # all translated as paths of landmarks to be saved
        for i in range(2, args):
            list_landm.append(sys.argv[i])
    else:
        list_landm = ['1_dlib_pred']


proportion = 0.2
figure_size = (10, 8)
overwrite = True
save_original = True
frames_format = '.png'


def for_each_landmark_folder(fold_landm):
    save_path = save_path_0 + fold_landm + '/'; mkdir_p(save_path)
    for clip in list_clips:
        path_frames = path_f + clip + '/'
        path_lndm = path_clips + fold_landm + '/' + clip + '/'
        if not check_path_and_landmarks(path_frames, clip, path_lndm):
            continue
        print(clip)
        save_path_i = save_path + clip + '/'; mkdir_p(save_path_i)
        imgs = generate_frames_max_bbox(path_frames, frames_format, [path_lndm],
                                        pts_format, [fold_landm],
                                        save_path_i, proportion, figure_size, overwrite, save_original, render_options)


save_path_0 = path_clips + foldvis + '/'; mkdir_p(save_path_0)
pts_format = ['_0' + pts_type_out]
path_f = path_clips + frames
list_clips = sorted(os.listdir(path_f))
frames_format = check_img_type(list_clips, path_f)


for landm in list_landm:
    if not check_if_path(path_clips + landm + '/', ''):
        continue
    for_each_landmark_folder(landm)

# fold_landm = '1_dlib_pred'
# for clip in list_clips:
#     path_frames = path_f + clip + '/'
#     path_lndm = path_clips + fold_landm + '/' + clip + '/'
#     if not check_path_and_landmarks(path_frames, clip, path_lndm):
#         continue
#     print(clip)
#     save_path_i = save_path + clip + '/'; mkdir_p(save_path_i)
#     imgs = generate_frames_max_bbox(path_frames, frames_format, [path_lndm],
#                                     pts_format, [fold_landm],
#                                     save_path_i, proportion, figure_size, overwrite, save_original, render_options)
