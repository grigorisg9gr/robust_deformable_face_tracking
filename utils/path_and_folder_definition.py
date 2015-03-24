__author__ = 'gchrysos'
# definition of paths and folders useful for the pipeline.
# Change with caution, as changes will be reflected to the whole pipeline.

__p_base_db = '/vol/atlas/databases/'

# paths of public databases used for trainings
path_to_helen = __p_base_db + 'helen/trainset/' # helen trainset
path_to_ibug = __p_base_db + 'ibug/'
path_to_lfpw = __p_base_db + 'lfpw/trainset/'
path_to_cofw = '/vol/atlas/homes/grigoris/external/cofw/frames/trainset/'
path_pascal_base = '/vol/atlas/homes/grigoris/external/VOCdevkit/VOC2007/'

path_pickles = '/vol/atlas/homes/grigoris/company_videos/pickles/'
path_shape_pred = '/vol/atlas/homes/grigoris/raps_menpo/shape_predictor_68_face_landmarks.dat' # predictor data trained to be used from dlib shape predictor

# confirm that the ones above are valid paths
from utils import check_if_path
def __db_p(path, db_name):
    return check_if_path(path, 'The database %s is not in the path provided (%s).' % (db_name, path))

if not (__db_p(path_to_helen, 'helen') and __db_p(path_to_ibug, 'ibug') and __db_p(path_pascal_base, 'pascal'))\
        and (__db_p(path_to_lfpw, 'lfpw')):
    print('Potential problem if one of the databases are not in the path provided.')


# folders for reading and writing in the project clips
foldvis = 'visualisations/' # folder where all image visualisations are
frames  = 'frames/'   # folder for reading the frames/images
foldcmp = 'compare/'  # folder for visual comparisons (will be inside visualisations by default)

visual = 0              # whether the landmarks should be exported during the process (1 for yes)
img_type_out = '.png'   # extension (and type) of the images that will be exported
pts_type_out = '.pts'

# definition of colours for visualisation
# OLD visualisations:
# import numpy as np
# red = np.array([[1], [0], [0]]); green = np.array([[0], [1], [0]]); blue = np.array([[0], [0], [1]])       # fix the color of the viewer (landmark points)
# oth_1 = np.array([[1], [1], [0]]); oth_2 = np.array([[1], [0], [1]]); oth_3 = np.array([[0], [1], [1]])
# colour = [red,green,blue, oth_1,oth_2,oth_3]; col_len = len(colour)
colour = ['r', 'b', 'g', 'c', 'm', 'k', 'w']
col_len = len(colour)


