

import matplotlib as mpl           # significant feature: For using the savefig in the python terminal. Should be added
mpl.use('Agg')                     # in the beginning of the program. http://stackoverflow.com/a/4935945/1716869


import sys, os
import menpo.io as mio
from utils import (mkdir_p, check_if_path)
from utils.pipeline_aux import (read_public_images, check_img_type, check_path_and_landmarks, load_images)
from utils.path_and_folder_definition import *  # import paths for databases, folders and visualisation options
import shutil

if __name__ == '__main__':
    args = len(sys.argv)
    if args>1:
        path_0 = str(sys.argv[1])
        if not(os.path.isdir(path_0)):
            raise RuntimeError('The path %s does not exist as base folder' % path_0)
    else:
        raise RuntimeError('file not called with initial path')

    if args > 2 and args < 6:
        in_landmarks_train_fol = str(sys.argv[2]) + '/'
        in_landmarks_test_fol = str(sys.argv[3]) + '/'
        out_landmarks_fol = str(sys.argv[4]) + '/'
        print in_landmarks_train_fol, '   ', in_landmarks_test_fol, '   ', out_landmarks_fol
    else:
        in_landmarks_train_fol = '1_dlib_pred/' # since we are confident that there are few false positives in dlib predictor
        in_landmarks_test_fol = '4_fit_pbaam/'
        out_landmarks_fol = '5_svm_faces/'


from menpo.feature import no_op, fast_dsift #double_igo
feat = fast_dsift
patch_s = (12, 12)
crop_reading = 0.2
pix_thres=130



path_init_sh = path_0 + in_landmarks_train_fol
path_clips = path_0 + frames
path_read_sh = path_0 + in_landmarks_test_fol
path_new_fit_vid = path_0 + foldvis + out_landmarks_fol; mkdir_p(path_new_fit_vid)
path_fitted_aam = path_0 + out_landmarks_fol; mkdir_p(path_fitted_aam)
path_pickle_svm = path_pickles + 'general_svm/'; mkdir_p(path_pickle_svm)
img_type     = '.png'
group_in     = 'In_detector'; group_out = 'shape_out'


import xml.etree.ElementTree as ET
def has_person(path):
    """ Returns true if the xml in the 'path' has a person in its annotation. Based on Pascal dataset xml files. """
    tree = ET.parse(path)
    root = tree.getroot()
    for key in range(6,len(root)):
        if root[key].findtext('name')=='person': 
            return True
    return False


import glob
def pascal_db_non_faces(base_path, path_faces, annot='Annotations/', im_fold='JPEGImages/', 
              training_images=[], max_loaded_images=None, pix_thres=150):
    ''' Loads the non-face images. Specifically it loads the images of PASCAL VOC Challenge 2007 and the 
        landmarks that are mistakenly predicted as faces by OpenCV (different script). It parses them based 
        on the list of non_faces that are false positives (OpenCV). It is much faster than the other version.'''
    # max_loaded_images -> in case we want to load up to max images for performance reasons
    list_non_faces = sorted(os.listdir(path_faces))
    im_path = base_path + im_fold; anno_path = base_path + annot
    if max_loaded_images==None: 
        max_loaded_images = len(list_non_faces)
    else: 
        max_loaded_images = min(len(list_non_faces), max_loaded_images)  # ensure that there are no more images asked than those that can be loaded
    cnt = 0; training_images = []
    for pts_f in list_non_faces:
        if cnt >= max_loaded_images: break
        im_n = pts_f[0:6]
        if has_person(anno_path + im_n + '.xml')==True: continue  #check that indeed there is no person in the annotations (which is the case if extracted with open cv code)
        res_im = glob.glob(im_path + im_n + '.*'); 
        if len(res_im)==1: 
            im = mio.import_image(res_im[0]); 
        else: continue
        ln = mio.import_landmark_file(path_faces + pts_f)
        im.landmarks['PTS'] =ln
        cnt +=1
        if im.n_channels == 3: im = im.as_greyscale(mode='luminosity')
        im.crop_to_landmarks_proportion_inplace(0.2)
        if im.shape[0]>pix_thres or im.shape[1]>pix_thres:
            im = im.rescale_to_diagonal(pix_thres)
        training_images.append(feat(im))
    return training_images, cnt



# Warping all images to a reference shape
from menpo.transform import PiecewiseAffine
from menpo.visualize import print_dynamic, progress_bar_str
from menpofit.aam.builder import build_reference_frame 
import numpy as np

def warp_all_to_reference_shape(images, meanShape):
    reference_frame = build_reference_frame(meanShape)
 
    # Build transforms for warping
    transforms = [PiecewiseAffine(reference_frame.landmarks['source'][None], i.landmarks['PTS'][None])
                          for i in images]
#     print transforms, '\n', transforms[0].n_dims
    # warp images to reference frame
    warped_images_list = []
    for c, (i, t) in enumerate(zip(images, transforms)):
        print_dynamic('Warping images: {}'.format(progress_bar_str(float(c + 1) / len(images), show_bar=True)))
        warped_images_list.append(i.warp_to_mask(reference_frame.mask, t))
    print('\rWarping images: Done')
    
    warped_images_ndarray = np.empty((warped_images_list[0].as_vector().shape[0], len(warped_images_list)))
    for j, i in enumerate(warped_images_list):
        i.landmarks['source'] = reference_frame.landmarks['source']
        warped_images_ndarray[:,j] = i.as_vector()
    warped_images_ndarray = warped_images_ndarray.T
    return warped_images_list, warped_images_ndarray, reference_frame

# same as function above, but for one image
def warp_image_to_reference_shape(i, reference_frame):
    transform = [PiecewiseAffine(reference_frame.landmarks['source'][None], i.landmarks['PTS'][None])]
    im2 = i.warp_to_mask(reference_frame.mask, transform[0])
    im2.landmarks['source'] = reference_frame.landmarks['source']
    return im2 #.as_vector()


#Extract patches around the images
# CASE 2: We want only the patches around landmarks
def list_to_nd_patches(images, patch_s=(12, 12)):
    s = images[0].extract_patches_around_landmarks(as_single_array=True, patch_size=patch_s).flatten().shape[0]
    arr = np.empty((len(images), s))
    for k,im in enumerate(images):
        _p_nd = im.extract_patches_around_landmarks(as_single_array=True, patch_size=patch_s)
        arr[k,:] = _p_nd.flatten()
    return arr



import matplotlib.pyplot as plt
import numpy as np
from random import randint
from joblib import Parallel, delayed
import glob

_c1 = colour[0]

def process_frame(frame_name, frames_path, pts_folder, path_all_clip, path_all_clip_2, clip_name, refFrame): 
    global clf
    # load image and the landmark points
    name = frame_name[:-4]; img_path = frames_path + name + img_type
    im = mio.import_image(img_path, normalise=True)
    if im.n_channels == 3: im = im.as_greyscale(mode='luminosity') 
    res = glob.glob(path_read_sh + clip_name + '/' + frame_name[:-4] + '*.pts')
    if len(res) == 0:
        return
    else:
        im_org = im  #im_org = im.copy(); im = feat(im); #keep a copy of the nimage if features in image level
        if visual == 1: rand = randint(1, 10000);  plt.figure(rand)
        for kk in range(0, len(res)):
            ln = mio.import_landmark_file(res[kk])
            im_cp = im.copy()
            im_cp.landmarks['PTS'] = ln
            im_cp.crop_to_landmarks_proportion_inplace(0.2);  im_cp = feat(im_cp) ############## ONLY IF THERE ARE 1,2 detections per image
            im2 = warp_image_to_reference_shape(im_cp, refFrame)
#             decision = clf.decision_function(im2.as_vector()) #case1
            _p_nd = im2.extract_patches_around_landmarks(as_single_array=True, patch_size=patch_s).flatten() #case2
            decision = clf.decision_function(_p_nd) #case2
#             print frame_name, '\t', kk, ' ', decision
            if decision < 0: p_s = path_all_clip_2
            else: 
                p_s = path_all_clip
                ending = res[kk].rfind('/') # find the ending of the filepath (the name of this landmark file)
                shutil.copy2(res[kk], pts_folder + res[kk][ending+1:])
            if visual == 1:
                plt.clf();
                if im_org.landmarks.has_landmarks == True: im_org.landmarks.clear()
                im_org.landmarks['PTS'] = ln
                if len(res) == 1: im_org.crop_to_landmarks_proportion_inplace(0.3)   ## zoom-in to see finer details for fitting, remove if unnecessary
                viewer = im_org.view_landmarks(group='PTS', render_numbering=False,  marker_face_colour=_c1, marker_edge_colour=_c1)
                # viewer = im_org.view_landmarks(group='PTS', render_numbering=False, lmark_view_kwargs={'colours': red})
                # viewer.figure.savefig(p_s + name + '_%d'%kk + img_type)
                viewer.save_figure(p_s + name + '_%d'%kk + img_type_out, pad_inches=0., overwrite=True, format=img_type_out[1::])
        if visual == 1: plt.close(rand) # plt.close('all') #problem with parallel # http://stackoverflow.com/a/21884375/1716869
            



def process_clip(clip_name, refFrame):
    print clip_name
    frames_path = path_clips + clip_name + '/'
    list_frames = sorted(os.listdir(frames_path))
    if not check_path_and_landmarks(frames_path, clip_name, path_read_sh + clip_name): # check that paths, landmarks exist
        return

    pts_folder = path_fitted_aam + clip_name + '/'; mkdir_p(pts_folder)
    path_all_clip = path_new_fit_vid + clip_name + '/'; mkdir_p(path_all_clip)
    path_all_clip_2 = path_new_fit_vid + clip_name + '_cut/'; mkdir_p(path_all_clip_2)
    # [process_frame(frame_name) for frame_name in list_frames];
    Parallel(n_jobs=-1, verbose=4)(delayed(process_frame)(frame_name, frames_path, pts_folder, path_all_clip,
                                                          path_all_clip_2, clip_name, refFrame) for frame_name in list_frames)


def load_or_train_svm():
    name_p = feat.__name__ + '_' + str(patch_s[0]) + '_' + str(crop_reading) + '_' + str(pix_thres) + '_' + 'helen_' + 'ibug'
    try:
        from sklearn.externals import joblib
        clf1 = joblib.load(path_pickle_svm + name_p + '.pkl')
        refFrame1 = joblib.load(path_pickle_svm + name_p + '_refFrame.pkl')
        return clf1, refFrame1
    except:         # if it doesn't exist, we should train the model
        # read the images from the public databases (ibug, helen)
        print('Loading Images ...')
        training_pos_im_1 = read_public_images(path_to_ibug, max_images=130, training_images=[], crop_reading=crop_reading, pix_thres=pix_thres, feat=feat)
        training_pos_im_1 = read_public_images(path_to_helen, max_images=220, training_images=training_pos_im_1, crop_reading=crop_reading, pix_thres=pix_thres, feat=feat)

        path_faces = path_pascal_base + 'grigoris_processing/' + '1_ocv_pred/'
        training_neg_im, cnt = pascal_db_non_faces(path_pascal_base, path_faces, max_loaded_images=500)

        # Procrustes analysis
        print('\nPerforming Procrustes analysis for alignment of the training images')
        from menpo.transform import GeneralizedProcrustesAnalysis as GPA
        gpa = GPA([i.landmarks['PTS'].lms for i in training_pos_im_1] )
        mean_shape = gpa.mean_aligned_shape()


        print('\nWarping the images and extracting patches')
        posim_l, _, refFrame1 = warp_all_to_reference_shape(training_pos_im_1, mean_shape)
        negim_l, _, refFrame1 = warp_all_to_reference_shape(training_neg_im, mean_shape)

        posim_nd = list_to_nd_patches(posim_l)
        negim_nd = list_to_nd_patches(negim_l)

        from sklearn import svm
        print('\nTraining SVM')
        # format data for sklearn svm
        tr_data = np.concatenate((posim_nd, negim_nd), axis=0)
        tr_class = np.zeros((tr_data.shape[0],))
        tr_class[0:posim_nd.shape[0]] = 1

        # train SVM
        clf1 = svm.LinearSVC(class_weight='auto', C=1).fit(tr_data, tr_class)
        _t = joblib.dump(clf1, path_pickle_svm + name_p + '.pkl')
        refFrame1 = joblib.dump(refFrame1, path_pickle_svm + name_p + '_refFrame.pkl')
        return clf1, refFrame1



clf, refFrame = load_or_train_svm()
list_done=[]
#list_done =['830386', '821238', '830844', '2Girls1Cup_crazy_reaction_1', '830183']; 
list_clips = sorted(os.listdir(path_clips))
img_type = check_img_type(list_clips, path_clips)  # assumption that all clips have the same extension, otherwise run in the loop for each clip separately.

[process_clip(clip_name, refFrame) for clip_name in list_clips
     if not((clip_name in list_done))];




