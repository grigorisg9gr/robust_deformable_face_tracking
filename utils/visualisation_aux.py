import matplotlib as mpl           # significant feature: For using the savefig in the python terminal. Should be added
mpl.use('Agg')                     # in the beginning of the program. http://stackoverflow.com/a/4935945/1716869
import matplotlib.pyplot as plt
from random import randint
import menpo.io as mio
import os
import numpy as np
import matplotlib.gridspec as gridspec
from menpo.image import Image


def _render(im, pts_names, fig, colours, markersizes, edgesizes, figure_size):
    if im.has_landmarks:
        for i, pts_name in enumerate(pts_names):
            if pts_name in im.landmarks.group_labels:
                renderer = im.view_landmarks(new_figure=False, figure_id=fig.number, group=pts_name,
                                             marker_edge_colour=colours[i],
                                             marker_face_colour=colours[i],
                                             marker_size=markersizes[i],
                                             marker_edge_width=edgesizes[i], figure_size=figure_size)
    else:
        renderer = im.view(new_figure=False, figure_id=fig.number, figure_size=figure_size)
    return renderer


def _aux(im, pts_paths, pts_names, pts_formats, save_path, save_original, off1, off2, figure_size, overwrite, render_options):
    # attach landmarks
    for k, pts_path in enumerate(pts_paths):
        if os.path.isfile(pts_path + im.path.stem + pts_formats[k]):
            pts = mio.import_landmark_file(pts_path + im.path.stem + pts_formats[k])
            im.landmarks[pts_names[k]] = pts

    # copy original if asked
    if save_original:
        im_orig = im.copy()

    # crop
    if pts_names[0] in im.landmarks.group_labels:
        centre = im.landmarks[pts_names[0]].lms.centre()
        min_indices = np.array([round(centre[0])-off1, round(centre[1])-off2])
        max_indices = np.array([round(centre[0])+off1, round(centre[1])+off2])
        im.crop_inplace(min_indices, max_indices)
    else:
        path_tmp = im.path
        im = Image.init_blank([off1*2 + 1, off2*2 + 1], im.n_channels)
        im.path = path_tmp

    # render
    rand = randint(1, 10000)
    fig = plt.figure(rand)
    if save_original:
        gs = gridspec.GridSpec(1, 2, width_ratios=[im_orig.height, im.height])

        plt.subplot(gs[0])
        renderer = _render(im_orig, pts_names, fig, render_options['colours'][0],
                           render_options['sizes'][0], render_options['edgesizes'][0], figure_size)

        plt.subplot(gs[1])
        renderer = _render(im, pts_names, fig, render_options['colours'][1],
                           render_options['sizes'][1], render_options['edgesizes'][1], figure_size)
    else:
        renderer = _render(im, pts_names, fig, render_options['colours'][1],
                           render_options['sizes'][1], render_options['edgesizes'][1], figure_size)

    renderer.save_figure(save_path + im.path.stem + '.png', format='png', pad_inches=0.0, overwrite=overwrite)
    plt.close(rand)


def generate_frames_max_bbox(frames_path, frames_format, pts_paths, pts_formats, pts_names, save_path,
                             proportion, figure_size, overwrite, save_original,
                             render_options, verbose=True):
    # find crop offset
    print('Computing max bounding box:')
    bounds_x = []
    bounds_y = []
    try:
        if len(os.listdir(pts_paths[0])) == 0:
            raise IndexError()
    except IndexError:
        if len(pts_paths) > 0:
            print('The directory of landmarks (%s) is empty, returning' % pts_paths[0])
        return
    for s in mio.import_landmark_files(pts_paths[0] + '*.pts', verbose=verbose):
        min_b, max_b = s.lms.bounds()
        bounds_x.append(max_b[0] - min_b[0])
        bounds_y.append(max_b[1] - min_b[1])
    off1 = round(max(bounds_x) * (1. + proportion) / 2)
    off2 = round(max(bounds_y) * (1. + proportion) / 2)

    print('\nLoad images, crop and save:')
    try:
        from joblib import Parallel, delayed
        Parallel(n_jobs=-1, verbose=4)(delayed(_aux)(im, pts_paths, pts_names, pts_formats, save_path, save_original,
                                                     off1, off2, figure_size, overwrite, render_options)
                                       for im in mio.import_images(frames_path + '*' + frames_format, verbose=False));
    except:
        print('Sequential execution')
        for im in mio.import_images(frames_path + '*' + frames_format, verbose=verbose):
            _aux(im, pts_paths, pts_names, pts_formats, save_path, save_original,
                 off1, off2, figure_size, overwrite, render_options);




