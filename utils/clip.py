from os.path import isdir, join


class Clip:
    """
    Fundamental info about a clip.
    :param clip_name:           Name of that clip.
    :param parent_path:         The parent folder, where all sub-folders for this clip are,
                                is used for defining the rest sub-folders.
    :param frames:              Name of the folder(s) from parent path to get to the clip.
    :param read_ln:             (optional) Name of folder(s) to read landmarks from.
    :param write_ln:            (optional) Name of folder(s) to write landmarks to, difference
                                from read_ln: No path is appended as prefix to this string/path.

    """
    def __init__(self, clip_name, parent_path, frames, read_ln=None, write_ln=None):
        assert(isinstance(clip_name, str))
        assert(isinstance(parent_path, str))
        assert(isinstance(read_ln, (type(None), str, list)))

        self.clip_name = clip_name
        self.parent_path = self.check_path(parent_path)
        self.path_frames = self.check_path(join(parent_path, frames, clip_name, ''))
        self.path_write_ln = write_ln
        # create the path_read_ln based on different formats of input.
        if isinstance(read_ln, str):
            read_ln = self.check_path(join(parent_path, read_ln, clip_name, ''))
            self.path_read_ln = read_ln
        elif isinstance(read_ln, list):  # in case of a list, append all paths
            self.path_read_ln = []
            for rl in read_ln:
                read_ln = self.check_path(join(parent_path, rl, clip_name, ''))
                self.path_read_ln.append(read_ln)


    def check_path(self, path):
        """
        Checks if a path exists. If it does, it returns the path, otherwise it raises an error.
        """
        if not(isdir(path)):
            raise RuntimeError('The path {} is not a valid path.'.format(path))
        return path
