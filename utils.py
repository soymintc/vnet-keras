import os
import glob
import keras
import nibabel
import time
from pathlib import Path
import numpy as np
from itertools import cycle
from numpy.random import random
from scipy.ndimage import interpolation
from collections import defaultdict
from scipy.ndimage import zoom
from keras import backend as K
from keras.callbacks import Callback


class ModelAndWeightsCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, jsonpath, monitor='val_loss', verbose=0,
                 save_best_only=False, 
                 mode='auto', period=1):
        super(ModelAndWeightsCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.jsonpath = jsonpath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            jsonpath = self.jsonpath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        self.model.save_weights(filepath, overwrite=True)
                        with open(jsonpath, 'w') as f:
                            f.write(self.model.to_json())
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                self.model.save_weights(filepath, overwrite=True)
                with open(jsonpath, 'w') as f:
                    f.write(self.model.to_json())


def add_midlines(data):
    assert isinstance(data, np.ndarray), "[ERROR] input image is not a np.array: {}".format(type(data))
    arr = data.copy()
    x_mid, y_mid, z_mid = np.median(np.array(([0]*3, data.shape)), axis=0).astype(int)
    max_val = np.max(arr)
    arr[x_mid-1:x_mid+1, :, :] = (max_val*0.2) * np.ones_like(arr[x_mid-1:x_mid+1, :, :])
    arr[:, y_mid-1:y_mid+1, :] = (max_val*0.5) * np.ones_like(arr[:, y_mid-1:y_mid+1, :])
    arr[:, :, z_mid-1:z_mid+1] = max_val * np.ones_like(arr[:, :, z_mid-1:z_mid+1])
    return arr


def dice_coefficient(y_true, y_pred, squared=True, smooth=1e-8):
    y_true_flat, y_pred_flat = K.flatten(y_true), K.flatten(y_pred)
    dice_nom = 2 * K.sum(y_true_flat * y_pred_flat)
    if squared:
        dice_denom = K.sum(K.square(y_true_flat) + K.square(y_pred_flat)) # squared form
    else:
        dice_denom = K.sum(K.abs(y_true_flat) + K.abs(y_pred_flat)) # abs form
    dice_coef = (dice_nom + smooth) / (dice_denom + smooth)
    return dice_coef
    
def dice_loss(y_true, y_pred, squared=True, smooth=1e-8):
    dice_coef = dice_coefficient(y_true, y_pred, squared, smooth)
    return 1 - dice_coef


class Transform3D(object):
    def __init__(self, rotation_range, shift_range, shear_range, zoom_range, flip, seed):
        np.random.seed(seed)
        
        self.rotation_angle = rotation_range * (random()-0.5)
        
        self.x_shift = shift_range * (random()-0.5) # 0.1 * 128 * [-1,+1]
        self.y_shift = shift_range * (random()-0.5) # 0.1 * 128 * [-1,+1]
        self.z_shift = shift_range * (random()-0.5) # 0.1 * 128 * [-1,+1]
        
        self.shear_matrix = np.array([[1, shear_range*(random()-0.5), shear_range*(random()-0.5)],
                                      [shear_range*(random()-0.5), 1, shear_range*(random()-0.5)],
                                      [shear_range*(random()-0.5), shear_range*(random()-0.5), 1]])
        
        self.zoom_factors = np.diag([zoom_range*(random()-0.5) for _ in range(3)])
        self.zoom_matrix = np.eye(3) - self.zoom_factors
        
        self.flip = flip
        self.flip_axis = []
        if self.flip:
            self.flip_axis = [x for x in range(3) if random()>0.5]
            
    def __repr__(self):
        return ("rotation_angle: {rotation_angle:.2f}, ".format(rotation_angle=self.rotation_angle) +
               "xyz-shifts: ({x_shift:.2f},{y_shift:.2f},{z_shift:.2f})\n".format(x_shift=self.x_shift, y_shift=self.y_shift, z_shift=self.z_shift) +
               "shear_matrix: {shear_matrix}\n".format(shear_matrix = np.round(self.shear_matrix, 2).tolist()) + 
               "zoom_factors: {zoom_factors}, ".format(zoom_factors = np.round(self.zoom_factors, 2).tolist()) +
               "flip_axis: {flip_axis}".format(flip_axis = self.flip_axis))
    
    def get_tag(self):
        r_tag = 'r{:.1f}'.format(self.rotation_angle)
        xyz_tag = 'xyz{:.0f},{:.0f},{:.0f}'.format(*np.array([self.x_shift, self.y_shift, self.z_shift])*100)
        f_tag = 'f{:s}'.format(str(self.flip_axis).replace('[','').replace(']','').replace(', ','')) 
        sz_tag = ['{:.0f}'.format(x) for x in 100*self.shear_matrix.dot(self.zoom_matrix).flatten()]
        sz_tag = 'sz' + ','.join(sz_tag)
        tag = '_'.join((r_tag, xyz_tag, f_tag, sz_tag))
        return tag
    
    
def transform_3d_array(arr, transform):
    """
    Return random (based on seed) transformation of 3D-array.
    Applies: rotation, shift, shear, zoom, and flip.
    Disclaimer: Tested only on 128x128x128 images.
    """
    
    #assert arr.shape == (128, 128, 128) # assume 128
    c_in = 0.5 * np.array(arr.shape)
    c_out = 0.5 * np.array(arr.shape)
    
    # Rotate: only using the z-axis (x,y-plane)
    rotated = interpolation.rotate(arr, transform.rotation_angle, axes=(0,1), order=0, reshape=False)
    
    # Shift
    x_L, y_L, z_L = rotated.shape # pixel lengths
    x_shift = x_L * transform.x_shift
    y_shift = y_L * transform.y_shift
    z_shift = z_L * transform.z_shift
    shift_offset = np.array([x_shift, y_shift, z_shift])
    
    # Shear
    shear_offset = c_in - c_out.dot(transform.shear_matrix)
    
    # Zoom
    zoom_offset = np.diag(np.array(arr.shape)/2 * transform.zoom_factors)
    
    # Shift + Shear + Zoom applied
    matrix = transform.zoom_matrix.dot(transform.shear_matrix)
    offset = shift_offset + shear_offset + zoom_offset
    transformed = interpolation.affine_transform(rotated, matrix, offset=offset, order=0)
    
    # Flip added
    if transform.flip:
        transformed = np.flip(transformed, axis=transform.flip_axis)
    
    return transformed


def fit_image_to_shape(arr, dst_shape=np.array([64,64,64]), order=0):
    return zoom(arr, np.divide(dst_shape, arr.shape), order=order)


def transform_and_save_data(transform, src_fpath, dst_dir, sample_id, tag, draw_midplanes=False):
    arr = nibabel.load(src_fpath).get_data()
    arr = fit_image_to_shape(arr)
    if draw_midplanes:
        arr = add_midlines(arr)
    arr = transform_3d_array(arr, transform)
    img = nibabel.Nifti1Image(arr, np.eye(4))
    dst_fname = '{}_{}_{}.nii.gz'.format(sample_id, transform.get_tag(), tag)
    dst_fpath = os.path.join(dst_dir, dst_fname)
    if os.path.exists(dst_fpath):
        print("[LOG] {} already exists. Will not save image".format(dst_fpath))
    else:
        nibabel.save(img, dst_fpath)
    
    
def augment_3d_data(src_dir, dst_dir, image_tags, label_tags,
                    rotation_range, shift_range, shear_range, zoom_range, flip, num_dst_samples,
                    dst_shape=(64,64,64), file_extension='nii.gz', draw_midplanes=False):
    
    assert os.path.isdir(src_dir), "[ERROR] {} is not a dir".format(src_dir)
    assert os.path.isdir(dst_dir), "[ERROR] {} is not a dir".format(dst_dir)
    tags = image_tags + label_tags
    
    sample_ids = set()
    image_files = defaultdict(dict)
    label_files = defaultdict(dict)
    
    fpaths = [x.as_posix() for x in Path(src_dir).rglob('*.'+file_extension)]
    for fpath in fpaths: # fpath: /path/to/SAMPLEID_seg|t1|t1ce|flair|t2.nii.gz
        fname = os.path.split(fpath)[-1]
        assert fname.endswith(file_extension), "[ERROR] {} does not end with {}".format(fname, file_extension)
        
        at_least_one_tag_in_fname = False
        for tag in tags:
            if tag in fname:
                at_least_one_tag_in_fname = True
                sample_id = fname[:fname.index(tag)-1]
                sample_ids.add(sample_id)
                if tag in image_tags:
                    image_files[sample_id][tag] = (fname, fpath)
                else:
                    label_files[sample_id][tag] = (fname, fpath)
        assert at_least_one_tag_in_fname, "[ERROR] No tags in {}".format(fname)
    
    sample_ids = sorted(sample_ids)# sort by alphabetical order
    num_samples_made = 0
    while num_dst_samples > num_samples_made:
        for ix, sample_id in enumerate(sample_ids):
            files_for_all_tags_made = False
            num_files_made_for_tag = 0
            seed = num_samples_made # set same seed for the tags of the same sample
            transform = Transform3D(rotation_range, shift_range, shear_range, zoom_range, flip, seed)

            for tag in image_tags:
                src_fname, src_fpath = image_files[sample_id][tag]
                transform_and_save_data(transform, src_fpath, dst_dir, sample_id, tag, draw_midplanes)
                num_files_made_for_tag += 1
            for tag in label_tags:
                src_fname, src_fpath = label_files[sample_id][tag]
                transform_and_save_data(transform, src_fpath, dst_dir, sample_id, tag, draw_midplanes)
                num_files_made_for_tag += 1
                
            assert num_files_made_for_tag % len(tags) == 0, "ERROR: Not all files for tags made for {}".format(sample_id)
            num_samples_made += 1
        # for sample end
        
        
class DataGenerator(keras.utils.Sequence):
    def __init__(self, ids, path, n_samples, batch_size=4, image_shape=(64,64,64), 
                rotation_range=0.2, shift_range=0.2, shear_range=0.2, zoom_range=0.2, flip=True):
        self.ids = ids
        self.path = path
        self.n_samples = n_samples # samples to create
        self.batch_size = batch_size
        #self.image_size = image_size
        #self.image_shape = (self.image_size,)*3
        self.image_shape = image_shape
        self.tids = [(name, True) for name in self.ids] # augment orig input as well
        
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.flip = flip

        if self.n_samples > len(self.ids): # if there's something to transform
            n_to_transform = self.n_samples - len(self.ids)
            for ix, name in enumerate(cycle(self.ids)):
                if ix < n_to_transform:
                    self.tids.append((name, True))
                else:
                    break
        
        self.on_epoch_end()
        
    def __load__(self, id_name, flag_transform): # load single image and label
        # Path
        t1_path = os.path.join(self.path, id_name + '_t1.nii.gz')
        t2_path = os.path.join(self.path, id_name + '_t2.nii.gz')
        t1ce_path = os.path.join(self.path, id_name + '_t1ce.nii.gz')
        flair_path = os.path.join(self.path, id_name + '_flair.nii.gz')
        seg_path = os.path.join(self.path, id_name + '_seg.nii.gz')
        assert os.path.exists(t1_path), "ERROR: {} does not exist".format(t1_path)
        assert os.path.exists(t2_path), "ERROR: {} does not exist".format(t2_path)
        assert os.path.exists(t1ce_path), "ERROR: {} does not exist".format(t1ce_path)
        assert os.path.exists(flair_path), "ERROR: {} does not exist".format(flair_path)
        assert os.path.exists(seg_path), "ERROR: {} does not exist".format(seg_path)
        
        # Read and concatenate normalized images
        t1 = fit_image_to_shape(nibabel.load(t1_path).get_data(), dst_shape=self.image_shape) / 255.
        t2 = fit_image_to_shape(nibabel.load(t2_path).get_data(), dst_shape=self.image_shape) / 255.
        t1ce = fit_image_to_shape(nibabel.load(t1ce_path).get_data(), dst_shape=self.image_shape) / 255.
        flair = fit_image_to_shape(nibabel.load(flair_path).get_data(), dst_shape=self.image_shape) / 255.
        #image = np.array([t1, t2, t1ce, flair]).transpose(1,2,3,0) # channels_last
        # image = np.array([flair, t1ce])#t1, t1ce, t2]) # channels_first
        image = np.array([flair, t1, t1ce, t2])#t1, t1ce, t2]) # channels_first # Fix #6
        
        # Transform if flag on
        if flag_transform:
            seed = np.random.randint(time.time() // 1000) # set same seed for the tags of the same sample
            transform = Transform3D(self.rotation_range, self.shift_range, 
                                    self.shear_range, self.zoom_range, self.flip, seed)
#             print(transform.get_tag(), id_name) ##@##
            for ix, arr in enumerate(image):
                image[ix] = transform_3d_array(arr, transform)
            seg = fit_image_to_shape(nibabel.load(seg_path).get_data(), dst_shape=self.image_shape)
            seg = transform_3d_array(seg, transform)
        else:
            seg = fit_image_to_shape(nibabel.load(seg_path).get_data(), dst_shape=self.image_shape)
            
        # Read and concatenate labels
        seg1 = (seg == 1) # tumor core
        seg2 = (seg == 2) # edema
        seg4 = (seg == 4) # enhancing
        #label = np.array([seg1, seg2, seg4]).transpose(1,2,3,0) # channels_last
        label = np.array([seg1, seg2, seg4]) # channels_first
        
        return image, label
    
    def __getitem__(self, ix): # load batch: batch_size*image, batch_size*label
        # Resize batch_size if overflown
        if (ix+1)*self.batch_size > len(self.tids): # when len(ids) % batch_size > 0
            self.batch_size = len(self.tids) - ix*self.batch_size
        
        files_batch = self.tids[ix*self.batch_size : 
                              (ix+1)*self.batch_size]
#         print("files_batch:", files_batch) ##@##
        images, labels = [], []
        for id_name, flag_transform in files_batch:
            # load img, label corresponding to the ID
            _image, _label = self.__load__(id_name, flag_transform)
            images.append(_image)
            labels.append(_label)
        images = np.array(images)
        labels = np.array(labels)
        
        return images, labels
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.tids) / float(self.batch_size)))

