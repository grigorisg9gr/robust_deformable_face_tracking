import os


class Clip:
    """
    Fundamental info about a clip.
    :param clip_name:           Name of that clip.
    :param parent_path:         The parent folder, where all subfolders for this clip are, will be used for defining the rest subfolders.
    :param frames:              Name of the folder(s) from parent path to get to the clip.
    :param read_ln:             (optional) Name of folder(s) to read landmarks from.
    :param write_ln:            (optional) Name of folder(s) to write landmarks to.

    """
    def __init__(self, clip_name, parent_path, frames, read_ln='', write_ln=''):
        self.clip_name = clip_name
        self.parent_path = self.check_path(parent_path)
        self.path_frames = self.check_path(parent_path + frames + clip_name + '/')
        if read_ln != '':
            read_ln = self.check_path(parent_path + read_ln + clip_name + '/')
        self.path_read_ln = read_ln
        self.path_write_ln = write_ln

    def check_path(self, path):
        """
        Checks if a path exists. If it does, it returns the path, otherwise it raises an error.
        """
        if not(os.path.isdir(path)):
            raise RuntimeError('The path %s is not a valid path' % path)
        return path
