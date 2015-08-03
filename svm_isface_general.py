import menpo.io as mio
from utils import (mkdir_p, print_fancy)
from utils.pipeline_aux import (read_public_images, check_img_type, im_read_greyscale,
                                check_path_and_landmarks, check_initial_path)
from utils.path_and_folder_definition import *  # import paths for databases, folders and visualisation options
import shutil
from joblib import Parallel, delayed

if __name__ == '__main__':
    args = len(sys.argv)
    path_0 = check_initial_path(args, sys.argv)

    if 2 < args < 5:
        in_landmarks_test_fol = str(sys.argv[2]) + sep
        out_landmarks_fol = str(sys.argv[3]) + sep
        print in_landmarks_test_fol, '   ', out_landmarks_fol
    else:
        in_landmarks_test_fol, out_landmarks_fol = '4_fit_pbaam' + sep, '5_svm_faces' + sep


from menpo.feature import no_op, fast_dsift, hog
feat = fast_dsift
patch_s = (14, 14)
crop_reading = 0.2
pix_thres = 170


path_clips = path_0 + frames
path_read_sh = path_0 + in_landmarks_test_fol
path_fitted_aam = mkdir_p(path_0 + out_landmarks_fol)
path_pickle_svm = mkdir_p(path_pickles + 'general_svm' + sep)

# Log file output.
log = mkdir_p(path_clips + 'logs' + sep) + datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + '_svm.log'
sys.stdout = Logger(log)


import xml.etree.ElementTree as ET
def has_person(path):
    """ Returns true if the xml in the 'path' has a person in its annotation. Based on Pascal dataset xml files. """
    tree = ET.parse(path)
    root = tree.getroot()
    for key in range(6, len(root)):
        if root[key].findtext('name') == 'person':
            return True
    return False


def pascal_db_non_faces(base_path, path_faces, annot='Annotations/', im_fold='JPEGImages/',
                        max_loaded_images=None, pix_thres=150):
    """
    Loads the non-face images of PASCAL VOC Challenge 2007 and the  landmarks that are mistakenly predicted as faces
    by OpenCV (different script). It parses them based  on the list of non_faces that are false positives (OpenCV).

    :param base_path:           Main path where the pascal files (images, annotations are).
    :param path_faces:          Path of the 'faces' as detected in PASCAL VOC.
    :param annot:               (optional) Folder name for original annotations' folder.
    :param im_fold:             (optional) Folder name for images.
    :param max_loaded_images:   (optional) Number of 'faces' to be loaded.
    :param pix_thres:           (optional) Pixel threshold for the dimensions.
    :return:
    """

    list_non_faces = sorted(os.listdir(path_faces))
    im_path = base_path + im_fold; anno_path = base_path + annot
    if max_loaded_images is None:
        max_loaded_images = len(list_non_faces)
    else: 
        max_loaded_images = min(len(list_non_faces), max_loaded_images)  # ensure that there are no more images asked than those that can be loaded
    cnt = 0; training_images = []
    for pts_f in list_non_faces:
        if cnt >= max_loaded_images:
            break
        im_n = pts_f[0:6]
        if has_person(anno_path + im_n + '.xml'):
            continue  # check that indeed there is no person in the annotations (which is the case if extracted with open cv code)
        res_im = glob.glob(im_path + im_n + '.*')
        if len(res_im) == 1:
            im = mio.import_image(res_im[0])
        else:
            continue
        ln = mio.import_landmark_file(path_faces + pts_f)
        im.landmarks['PTS'] = ln
        cnt +=1
        if im.n_channels == 3: im = im.as_greyscale(mode='luminosity')
        im.crop_to_landmarks_proportion_inplace(0.2)
        if im.shape[0] > pix_thres or im.shape[1] > pix_thres:
            im = im.rescale_to_diagonal(pix_thres)
        training_images.append(feat(im))
    return training_images, cnt



# Warping all images to a reference shape
from menpo.transform import PiecewiseAffine
from menpo.visualize import print_dynamic, progress_bar_str
from menpofit.aam.builder import build_reference_frame

def warp_all_to_reference_shape(images, meanShape):
    reference_frame = build_reference_frame(meanShape)
 
    # Build transforms for warping
    transforms = [PiecewiseAffine(reference_frame.landmarks['source'][None], i.landmarks['PTS'][None])
                  for i in images]
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
    return im2


# Extract patches around each landmark of the image
def list_to_nd_patches(images, patch_s=(12, 12)):
    s = images[0].extract_patches_around_landmarks(as_single_array=True, patch_size=patch_s).flatten().shape[0]
    arr = np.empty((len(images), s))
    for k,im in enumerate(images):
        _p_nd = im.extract_patches_around_landmarks(as_single_array=True, patch_size=patch_s)
        arr[k, :] = _p_nd.flatten()
    return arr


def process_frame(frame_name, frames_path, pts_folder, clip_name, refFrame):
    global clf
    # load image and the landmark points
    im = im_read_greyscale(frame_name, frames_path, img_type)
    if not im:
        return
    res = glob.glob(path_read_sh + clip_name + sep + im.path.stem + '*.pts')
    im_org = im  #im_org = im.copy(); im = feat(im); #keep a copy of the nimage if features in image level
    # print im.path.stem, len(res)
    for kk, ln_n in enumerate(res):
        ln = mio.import_landmark_file(ln_n)
        im_cp = im.copy()
        im_cp.landmarks['PTS'] = ln
        # im_cp.crop_to_landmarks_proportion_inplace(0.2);  im_cp = feat(im_cp) ############## ONLY IF THERE ARE 1,2 detections per image
        im_cp = im_cp.crop_to_landmarks_proportion(0.2); im_cp = feat(im_cp) ############## ONLY IF THERE ARE 1,2 detections per image
        im2 = warp_image_to_reference_shape(im_cp, refFrame)
        _p_nd = im2.extract_patches_around_landmarks(as_single_array=True, patch_size=patch_s).flatten()
        decision = clf.decision_function(_p_nd)
        if decision > 0:
            ending = ln_n.rfind(sep)  # find the ending of the filepath (the name of this landmark file)
            shutil.copy2(ln_n, pts_folder + ln_n[ending+1:])


def process_clip(clip_name, refFrame):
    print clip_name
    frames_path = path_clips + clip_name + sep
    list_frames = sorted(os.listdir(frames_path))
    if not check_path_and_landmarks(frames_path, clip_name, path_read_sh + clip_name): # check that paths, landmarks exist
        return

    pts_folder = mkdir_p(path_fitted_aam + clip_name + sep)
    #[process_frame(frame_name, frames_path, pts_folder,clip_name, refFrame) for frame_name in list_frames];
    Parallel(n_jobs=-1, verbose=4)(delayed(process_frame)(frame_name, frames_path, pts_folder,
                                                          clip_name, refFrame) for frame_name in list_frames)


def load_or_train_svm():
    name_p = feat.__name__ + '_' + str(patch_s[0]) + '_' + str(crop_reading) + '_' + str(pix_thres) + '_' + 'helen_' + 'ibug_' + 'lfpw'
    try:
        from sklearn.externals import joblib
        clf1 = joblib.load(path_pickle_svm + name_p + '.pkl')
        refFrame1 = joblib.load(path_pickle_svm + name_p + '_refFrame.pkl')
        return clf1, refFrame1
    except:         # if it doesn't exist, we should train the model
        # read the images from the public databases (ibug, helen)
        print('Loading Images')
        training_pos_im_1 = read_public_images(path_to_ibug, max_images=130, training_images=[], crop_reading=crop_reading, pix_thres=pix_thres, feat=feat)
        training_pos_im_1 = read_public_images(path_to_helen, max_images=420, training_images=training_pos_im_1, crop_reading=crop_reading, pix_thres=pix_thres, feat=feat)
        training_pos_im_1 = read_public_images(path_to_lfpw, max_images=100, training_images=training_pos_im_1, crop_reading=crop_reading, pix_thres=pix_thres, feat=feat)

        path_faces = path_pascal_base + 'grigoris_processing/' + '1_ocv_pred/'
        training_neg_im, cnt = pascal_db_non_faces(path_pascal_base, path_faces, max_loaded_images=600)

        # Procrustes analysis
        print('\nPerforming Procrustes analysis for alignment of the training images')
        from menpo.transform import GeneralizedProcrustesAnalysis as GPA
        gpa = GPA([i.landmarks['PTS'].lms for i in training_pos_im_1] )
        mean_shape = gpa.mean_aligned_shape()

        print('\nWarping the images and extracting patches')
        posim_l, _, refFrame1 = warp_all_to_reference_shape(training_pos_im_1, mean_shape)
        negim_l, _, refFrame1 = warp_all_to_reference_shape(training_neg_im, mean_shape)

        posim_nd = list_to_nd_patches(posim_l, patch_s)
        negim_nd = list_to_nd_patches(negim_l, patch_s)

        from sklearn import svm
        print('\nTraining SVM')
        # format data for sklearn svm
        tr_data = np.concatenate((posim_nd, negim_nd), axis=0)
        tr_class = np.zeros((tr_data.shape[0],))
        tr_class[0:posim_nd.shape[0]] = 1

        # train SVM
        clf1 = svm.LinearSVC(class_weight='auto', C=1).fit(tr_data, tr_class)
        _t = joblib.dump(clf1, path_pickle_svm + name_p + '.pkl')
        _t2 = joblib.dump(refFrame1, path_pickle_svm + name_p + '_refFrame.pkl')
        return clf1, refFrame1


print_fancy('Training SVMs trained on public datasets')
clf, refFrame = load_or_train_svm()
list_clips = sorted(os.listdir(path_clips))
img_type = check_img_type(list_clips, path_clips)  # assumption that all clips have the same extension, otherwise run in the loop for each clip separately.

[process_clip(clip_name, refFrame) for clip_name in list_clips
 if not(clip_name in list_done)]




