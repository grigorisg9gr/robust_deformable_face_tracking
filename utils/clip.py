from os.path import isdir, join


class Clip:
    """
    Fundamental info about a clip.
    :param clip_name:           str: Name of the clip.
    :param parent_path:         str: Base path that contains the frames/lns folders.
    :param frames:              str: Folder name for the base folder of the frames.
    :param read_ln:             list/None: (optional) Name of folder(s) to read landmarks from.
    :param write_ln:            str/list/None: (optional) Name of folder(s) to write landmarks to, difference
                                from read_ln: No path is appended as prefix to this string/path.
    """
    def __init__(self, clip_name, parent_path, frames, read_ln=None, write_ln=None):
        assert(isinstance(clip_name, str))
        assert(isinstance(parent_path, str))
        assert(isinstance(read_ln, (type(None), list)))

        self.clip_name = clip_name
        self.parent_path = self.check_path(parent_path)
        self.path_frames = self.check_path(join(parent_path, frames, clip_name, ''))
        self.path_write_ln = write_ln
        # create the path_read_ln as a list of paths.
        if isinstance(read_ln, list):
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
