import sys, os
if __name__ == '__main__':
    args = len(sys.argv)
    if args > 1:
        path_clips = str(sys.argv[1])
    else:
        raise RuntimeError('file not called with initial path')

    list_to_check = []
    if args > 2: # all translated as paths of landmarks to be saved
        for i in range(2, args):
            list_to_check.append(sys.argv[i])
    else:
        list_to_check = ['1_dlib_pred', '2_dpm', '4_fit_pbaam', '5_svm_faces']

sys.path.append('/vol/atlas/homes/grigoris/auxiliary/python_tools/more/'); from send_mail import * 
#from datetime import datetime
#send_mail(message=path_clips + '\n\n' + str(datetime.now()))


def not_a_path(path):
    '''Checks whether a given path is a directory in the current system. If it's not, it raises a value error.'''
    import os
    if not(os.path.isdir(path)):
        raise ValueError('The path %s is not a valid folder.\n' %path)


def path_checks(path_folders, frames_fold):
    not_a_path(path_folders) 
    not_a_path(frames_fold)
    number_clips = len(os.listdir(frames_fold))
    if number_clips == 0: 
        raise ValueError('There are no clips in the path %s.\n' %frames_fold)
    return number_clips

def check_progress(path_folders, message=''):
    import os, sys
    message += path_folders + '  :\n\n'
    frames_fold = path_folders + 'frames/'  # the folder/path where the initial frames are. 
    number_clips = len(os.listdir(frames_fold))
    number_clips = path_checks(path_folders, frames_fold)
    list_folders = sorted(os.listdir(path_folders))

    for folder in list_to_check:
        if folder in list_folders:
            number_subfolders = len(os.listdir(path_folders + folder + '/'))
            message += '%s : %d subfolders (out of %d clips).\n' % (folder, number_subfolders, number_clips)
        else: 
            message += 'The file %s does not exist.\n' %folder
    return message


send_mail(message=check_progress(path_clips))


