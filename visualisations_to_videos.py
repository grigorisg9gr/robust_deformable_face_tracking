
from os import listdir
from os.path import join, sep
import sys
from utils import (check_if_path, remove_empty_folders)


# import the function that reads frames and converts them to videos
#sys.path.append('/vol/atlas/homes/grigoris/gits/pyutils/')
#import frames2videos as fr2vid
#import resize_image as rim
# requires the package from https://github.com/grigorisg9gr/pyutils
from research_pyutils.frames2videos import main as fr2vid_main
from research_pyutils.resize_image import bulkResize


def main_call_visualisation_to_videos(path_0, overwrite):
    path_clips = join(path_0, 'visualisations', '')

    if not check_if_path(path_clips, ''):
            raise ValueError('The visualisation path {} does not seem to exist'.format(path_clips))

    # call from parent folder of visualisations, it will make videos for all different steps of the algorithm
    vid_fold = '1_videos'
    call_video_maker(path_clips, vid_fold, overwrite)
    p_cmp = join(path_clips, 'compare', '')
    if check_if_path(p_cmp, ''):  # call for comparisons as well
        call_video_maker(p_cmp, vid_fold, overwrite)


def call_video_maker(path_clips, vid_fold, overwrite=False):
    list_paths = sorted(listdir(path_clips))
    for i in list_paths:
        if i == 'compare':
            continue
        p1 = path_clips + i + sep
        if not check_if_path(p1, ''):  # if the folder does not exist, continue.
            continue
        # if: a) it has sufficient files, b) overwrite == False, then continue.
        if (not overwrite) and check_if_path(p1 + vid_fold, ''):
            len_clips = len(listdir(p1))
            print(i, '  ', len_clips, '  ', len(listdir(p1 + vid_fold)))
            if len(listdir(p1 + vid_fold)) > len_clips-3:
                continue

        remove_empty_folders(p1)
        print(i, '  ', len(listdir(p1)))
        if len(listdir(p1)) == 0:
            continue
        try:
            bulkResize(p1)
        except TypeError as e:
            print('Probably there is a folder with no images {}, skipping it.'.format(p1))
            print(e)
            continue
        except IOError as e:
            print('Probably image not found, skipping this video.')
            print(e)
            continue
        fr2vid_main(p1, vid_fold=vid_fold)
    remove_empty_folders(path_clips)


if __name__ == '__main__':
    overwrite_m = False  # Overwrite old written videos
    args = len(sys.argv)
    if args > 1:
        path_0_m = str(sys.argv[1])
    else:
        raise RuntimeError('File not called with initial path.')
    if args > 2:
        overwrite_m = True
    main_call_visualisation_to_videos(path_0_m, overwrite_m)
