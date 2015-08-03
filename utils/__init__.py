

import os
import sys
import shutil
import errno

sep = os.path.sep

def mkdir_p(path):
    """
    'mkdir -p' in Python from http://stackoverflow.com/a/11860637/1716869
    It creates all the subfolders till the end folder.
    """
    try:
        os.makedirs(path)
        return path
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return path
        else:
            raise

def rm_if_exists(path):
    try:
        shutil.rmtree(path)
    except OSError:
        pass


def check_if_path(path, msg='Not valid path'):
    """
    Checks if a path exists.
    If it does, it returns True, otherwise it prints a message (msg) and returns False
    """
    if not(os.path.isdir(path)):
        print(msg)
        return False
    return True


def find_image_type(dirname, fname):
    """
    Accepts a directory and a filename and returns the extension of the image. The image should
    have one of the extensions listed below, otherwise an exception is raised.
    """
    from os.path import splitext
    extensions = {'png', 'jpg', 'gif', 'jpeg'}  # can add to this list if required
    try:
        image_type = splitext(fname)[-1][1:]  # Assumption that the first element is an image.
    except IOError:
        print('Probably the first element in the folder is not an image, which is required')
        raise
    if image_type in extensions:
        return image_type
    else:
        import imghdr
        try:
            type1 = imghdr.what(dirname + sep + fname)
        except IOError:
            raise IOError('The file %s does not exist.\n' % (dirname + sep + fname))
        if type1 in extensions:
            return type1
        else:
            raise ValueError('%s is not supported type (extension) of image' % type1)


def remove_empty_folders(path):
    """
    Accepts a path/directory and removes all empty folders in that path (not the empty ones in the subfolders).
    """
    if not check_if_path(path, 'The path {} is not valid.'.format(path)):
        return -1
    sub_folders = os.listdir(path)
    for fol in sub_folders:
        p1 = path + fol + sep
        if not check_if_path(p1, ''):  # then it is not a folder
            continue
        if len(os.listdir(p1)) == 0:
            print('The folder {} is empty, removing.'.format(fol))
            rm_if_exists(p1)


def print_fancy(str1, str_after='\n'):
    """
    Function that prints the selected message (str1) surrounded by lines.
    :param str1:            The main message that will be printed.
    :param str_after:       (optional) Print empty lines after the main message.
    :return:
    """
    len_str = len(str1)
    if len(str1) <= 0:
        pass
    else:
        len_str += 4
        horiz_line = ['-']*len_str
        s = ''.join(horiz_line)
        print(s)
        print('| ' + str1 + ' |')
        print(s)
    print(str_after)


class Logger(object):
    # Log files -> http://stackoverflow.com/a/5916874/1716869
    def __init__(self, filename='Default.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)


