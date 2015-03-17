
import os,sys
from utils import (check_if_path, remove_empty_folders)

overwrite = False # Overwrite old written videos
if __name__ == '__main__':
    args = len(sys.argv)
    if args > 1:
        path_clips = str(sys.argv[1]) + '/visualisations/'
    else:
        raise RuntimeError('file not called with initial path')
    if args > 2:
        overwrite = True




# import the function that reads frames and converts them to videos
sys.path.append('/vol/atlas/homes/grigoris/gits/pyutils/')
import from_frames_to_videos as fr2vid
import resize_image as rim

if not check_if_path(path_clips, 'The visualisation path (%s) does not seem to exist' % path_clips):
        exit()



def call_video_maker(path_clips, vid_fold):
    list_paths = sorted(os.listdir(path_clips))
    for i in list_paths:
        if i == 'compare':
            continue
        p1 = path_clips + i + '/'
        # if: a) the video folder exists, b) it has sufficient files, c) overwrite == False, then continue
        if check_if_path(p1) and (not overwrite) and check_if_path(p1 + vid_fold, ''):
            len_clips = len(os.listdir(p1))
            print i, '  ', len(os.listdir(p1)), '  ', len(os.listdir(p1 + vid_fold))
            if len(os.listdir(p1 + vid_fold)) > len_clips-3:
                continue
        remove_empty_folders(p1)
        print i, '  ', len(os.listdir(p1))
        rim.bulkResize(path_clips + i + '/')
        fr2vid.main(path_clips + i + '/', vid_fold=vid_fold)


# call from parent folder of visualisations, it will make videos for all different steps of the algorithm
vid_fold = '1_videos'
call_video_maker(path_clips, vid_fold)
if check_if_path(path_clips + 'compare/', ''):  # call for comparisons as well
    call_video_maker(path_clips + 'compare/', vid_fold)