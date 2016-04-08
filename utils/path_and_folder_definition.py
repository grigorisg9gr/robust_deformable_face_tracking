from os.path import sep, isdir
# definition of paths and folders useful for the pipeline.
# Change with caution, as changes will be reflected to the whole pipeline.


__p_base_db = '/vol/atlas/databases/'
_p_base_personal = '/vol/atlas/homes/grigoris/'

# paths of public databases used for trainings
path_to_helen = __p_base_db + 'helen/trainset/'  # helen trainset
path_to_ibug = __p_base_db + 'ibug/'
path_to_lfpw = __p_base_db + 'lfpw/trainset/'
path_to_cofw = _p_base_personal + 'external/cofw/frames/trainset/'
path_pascal_base = _p_base_personal + 'external/VOCdevkit/VOC2007/'
path_closed_eyes = _p_base_personal + 'Databases/eyes/grigoris_competition_8_2015/frames/'

path_pickles = _p_base_personal + 'company_videos/pickles/'
# predictor data trained to be used from dlibERT shape predictor
path_shape_pred = _p_base_personal + 'raps_menpo/shape_predictor_68_face_landmarks.dat'

# confirm that the ones above are valid paths

def __db_p(path, db_name):
    dec = isdir(path)
    if not dec:
        print('The database {} is not in the path provided ({}).'.format(db_name, path))
    return dec

if not (__db_p(path_to_helen, 'helen') and __db_p(path_to_ibug, 'ibug') and __db_p(path_pascal_base, 'pascal'))\
        and (__db_p(path_to_lfpw, 'lfpw')):
    print('Potential problem if one of the databases are not in the path provided.')


# folders for reading and writing in the project clips
foldvis = 'visualisations' + sep  # folder where all image visualisations are
frames  = 'frames' + sep  # folder for reading the frames/images
foldcmp = 'compare' + sep  # folder for visual comparisons (will be inside visualisations by default)

visual = 0              # whether the landmarks should be exported during the process (1 for yes)
img_type_out = '.png'   # extension (and type) of the images that will be exported
pts_type_out = '.pts'

list_done = []            # clips that should not be processed
# list_done =['830386', '821238', '830183'];

# definition of colours for visualisation
colour = ['r', 'b', 'g', 'c', 'm', 'k', 'w']
col_len = len(colour)

# First refers to original image, second to cropped one.
render_options = {'colours':    [colour,
                                 colour],
                  'sizes':      [[2]*10,
                                 [2]*10],
                  'edgesizes':  [[1]*10,
                                 [2]*10]}


# common imports for all files
import os
import sys
import numpy as np
import glob
from datetime import datetime

