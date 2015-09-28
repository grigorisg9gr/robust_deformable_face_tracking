
import os
import sys
from utils import (check_if_path, remove_empty_folders)
sep = os.path.sep

# import the function that reads frames and converts them to videos
sys.path.append('/vol/atlas/homes/grigoris/gits/pyutils/')
import frames2videos as fr2vid
import resize_image as rim


def main_call_visualisation_to_videos(path_0, overwrite):
    path_clips = path_0 + sep + 'visualisations' + sep

    if not check_if_path(path_clips, ''):
            raise ValueError('The visualisation path {} does not seem to exist'.format(path_clips))

    # call from parent folder of visualisations, it will make videos for all different steps of the algorithm
    vid_fold = '1_videos'
    call_video_maker(path_clips, vid_fold, overwrite)
    if check_if_path(path_clips + 'compare' + sep, ''):  # call for comparisons as well
        call_video_maker(path_clips + 'compare' + sep, vid_fold, overwrite)


def call_video_maker(path_clips, vid_fold, overwrite=False):
    list_paths = sorted(os.listdir(path_clips))
    for i in list_paths:
        if i == 'compare':
            continue
        p1 = path_clips + i + sep
        # if: a) the video folder exists, b) it has sufficient files, c) overwrite == False, then continue
        if check_if_path(p1, '') and (not overwrite) and check_if_path(p1 + vid_fold, ''):
            len_clips = len(os.listdir(p1))
            print(i, '  ', len(os.listdir(p1)), '  ', len(os.listdir(p1 + vid_fold)))
            if len(os.listdir(p1 + vid_fold)) > len_clips-3:
                continue
        if not check_if_path(p1, ''):
            continue
        remove_empty_folders(p1)
        print(i, '  ', len(os.listdir(p1)))
        if len(os.listdir(p1)) == 0:
            continue
        try:
           rim.bulkResize(path_clips + i + sep)
        except TypeError as e:
           print('Probably there is a folder with no images {}, skipping it.'.format(path_clips + i + sep))
           print(e)
           continue
        except IOError as e:
           print('Probably image not found, skipping this video.')
           print(e)
           continue
        fr2vid.main(path_clips + i + sep, vid_fold=vid_fold)
    remove_empty_folders(path_clips)


if __name__ == '__main__':
    overwrite_m = False  # Overwrite old written videos
    args = len(sys.argv)
    # path_0_m = '/vol/atlas/homes/grigoris/company_videos/external/stefanos_3_septem/'
    if args > 1:
        path_0_m = str(sys.argv[1])
    else:
        raise RuntimeError('File not called with initial path.')
    if args > 2:
        overwrite_m = True
    main_call_visualisation_to_videos(path_0_m, overwrite_m)
