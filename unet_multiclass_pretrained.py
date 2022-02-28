import os
import logging
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import random
import tensorflow as tf

import glob
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mse
from tensorflow.keras.regularizers import l2, l1
import gc
from tensorflow import keras
import matplotlib.pyplot as plt
from keras_unet_collection import models
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras import backend as K

from pathlib import Path

from tqdm import tqdm
import cv2

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop
)


# def dice_coef_9cat(y_true, y_pred, smooth=1e-7):
#     '''
#     Dice coefficient for 10 categories. Ignores background pixel label 0
#     Pass to model as metric during compile statement
#     '''
#     y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=4)[...,1:])
# #     y_true_f = K.flatten(y_true[...,1:])
#     y_pred_f = K.flatten(y_pred[...,1:])
#     intersect = K.sum(y_true_f * y_pred_f, axis=-1)
#     denom = K.sum(y_true_f + y_pred_f, axis=-1)
#     return 1- K.mean((2. * intersect / (denom + smooth)))


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, problem_type):
    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = binary_factor
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = parts_factor
    elif problem_type == 'instruments':
        factor = instrument_factor
        mask_folder = 'instruments_masks'

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)

    return (mask / factor).astype(np.uint8)


def get_split(fold):
    folds = {0: [1],
             1: [2, 5],
             2: [4, 8],
             3: [6, 7]}

    train_path = data_path / 'cropped_train'

    train_file_names = []
    val_file_names = []
    test_file_names = []

    for instrument_id in range(1, 9):
        if instrument_id in folds[fold]:
            val_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))
        elif instrument_id==3:
            test_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))
        else:
            train_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))

    return train_file_names, val_file_names,test_file_names


def train_transform(p=1):
        return Compose([
            PadIfNeeded(min_height=1024, min_width=1280, p=1),
            RandomCrop(height=1024, width=1280, p=1),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            Normalize(p=1)
        ], p=p)

def val_transform(p=1):
    return Compose([
        PadIfNeeded(min_height=1024, min_width=1280, p=1),
        CenterCrop(height=1024, width=1280, p=1),
        Normalize(p=1)
    ], p=p)

def aug_fu_val(image,label):
    v_transform = val_transform()
    data = {"image":image,"label":label}
    aug_data = v_transform(**data)
    aug_img = aug_data["image"]
    aug_label = aug_data["label"]
    return aug_img,aug_label


def aug_fu_train(image,label):
    transform = train_transform()
    data = {"image":image,"label":label}
    aug_data = transform(**data)
    aug_img = aug_data["image"]
    aug_label = aug_data["label"]
    return aug_img,aug_label

def preprocess_train_data(images_T,labels_T):
    aug_imgs_T = np.zeros(images_T.shape)
    aug_labels_T = np.zeros(labels_T.shape)
    for i in range(len(images_T)):
        aug_imgs_T[i], aug_labels_T[i]=aug_fu_train(images_T[i],labels_T[i])
    return aug_imgs_T,aug_labels_T


def preprocess_val_data(images_V,labels_V):
    aug_imgs_V = np.zeros(images_V.shape)
    aug_labels_V = np.zeros(labels_V.shape)
    for i in range(len(images_V)):
        aug_imgs_V[i], aug_labels_V[i]=aug_fu_val(images_V[i],labels_V[i])
    return aug_imgs_V,aug_labels_V

# def IOU_CE_loss(y_true, y_pred):
#     y_pred = K.batch_flatten(y_pred)
#     y_true = K.batch_flatten(y_true)

#     _bce_loss_fu = tf.keras.losses.BinaryCrossentropy()
#     ce_loss = _bce_loss_fu(y_true, y_pred)
    
    
#     class_intersection = K.sum(y_true * y_pred)
#     class_loss = (class_intersection + 1) / (K.sum(y_true) + K.sum(y_pred) - class_intersection + 1)
   
    
#     return 0.7*ce_loss - 0.3*tf.math.log(class_loss)

def IOU_CCE_loss(y_true, y_pred):


    y_true = K.one_hot(K.cast(y_true, 'int32'), num_classes=4)
    _cce_loss_fu = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    cce_loss = _cce_loss_fu(y_true,y_pred)
    # print(cce_loss)
    loss = 0.7 * cce_loss
    
    for cls_index in range(4):
        y_pred_single = K.flatten(y_pred[...,cls_index])
        y_true_single = K.flatten(y_true[...,cls_index])

        class_intersection = K.sum(y_true_single * y_pred_single)
        class_loss = (class_intersection + 1) / (K.sum(y_true_single) + K.sum(y_pred_single) - class_intersection + 1)
        _iou_loss = 0.3*tf.math.log(class_loss)
        loss = loss- _iou_loss
   
    # print(cce_loss,loss)
    return loss





#  Swin-UNET for 3-label classification with:

# * input size of (128, 128, 3)

# * Four down- and upsampling levels (or three downsampling levels and one bottom level) (`depth=4`).

# * Two Swin-Transformers per downsampling level.

# * Two Swin-Transformers (after concatenation) per upsampling level.

# * Extract 2-by-2 patches from the input (`patch_size=(2, 2)`)

# * Embed 2-by-2 patches to 64 dimensions (`filter_num_begin=64`, a.k.a, number of embedded dimensions).

# * Number of attention heads for each down- and upsampling level: `num_heads=[4, 8, 8, 8]`.

# * Size of attention windows for each down- and upsampling level: `window_size=[4, 2, 2, 2]`.

# * 512 nodes per Swin-Transformer (`num_mlp=512`)

# * Shift attention windows (i.e., Swin-MSA) (`shift_window=True`).

# model = models.swin_unet_2d((128, 128, 3), filter_num_begin=64, n_labels=3, depth=4, stack_num_down=2, stack_num_up=2, 
#                         patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512, 
#                         output_activation='Softmax', shift_window=True, name='swin_unet')







# def train_model_kd(total_epochs,save_dir,images,labels):
def train_model(total_epochs, save_dir,train_images,train_labels,val_images,val_labels):


    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename='parts_unet.log', level=logging.DEBUG, format=LOG_FORMAT)







    model = models.unet_2d((1024, 1280,3), [64, 128, 256, 512], n_labels=4,
                      stack_num_down=1, stack_num_up=1,
                      activation='GELU', output_activation='Softmax', 
                      batch_norm=True, pool='max', unpool='nearest', name='unet')
    
    model.load_weights("./pretrained_model/model_unet3D_parts_origin.h5")

    

    model.compile(loss=IOU_CCE_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad = True))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    nEpochs = total_epochs
    batchSize = 1

    tbEpoch = 0

    num_sample = train_images.shape[0]
    sub_num = 1
    sub_index = [np.int32(num_sample*s/sub_num) for s in range(sub_num+1)  ]

    min_val_loss = 10000.0
    
    aug_img_val, aug_label_val = preprocess_val_data(val_images,val_labels)

            
    for epoch in range(nEpochs):
    #     display.clear_output()
        data_list = np.arange(num_sample)
        np.random.shuffle(data_list)
        if epoch==10:
            model.compile(loss=IOU_CCE_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001, amsgrad = True))
    
        for sub in range(sub_num):  


            tmp_image = train_images[data_list[sub_index[sub]:sub_index[sub+1]],:,:,:].astype(np.float32)
            tmp_label = train_labels[data_list[sub_index[sub]:sub_index[sub+1]],:,:].astype(np.float32)
            tmp_image, tmp_label = preprocess_train_data(tmp_image,tmp_label)
  
            history = model.fit(tmp_image , tmp_label, 
#                                 validation_data=(valid_img, valid_label),
                                batch_size=batchSize, 
                                epochs = tbEpoch + 1, 
                                initial_epoch = tbEpoch, 
                                shuffle = False, 
                                verbose = 1)  
            tbEpoch += 1
            
#             s_predict = model.predict(aug_img_val.astype(np.float32),batch_size=1)
#             val_loss = IOU_CCE_loss(aug_label_val.astype(np.float32),s_predict).numpy()
  
            
#             _suffix = tbEpoch % 2
#             s_unet.save_weights(save_dir + "model_unet3D_distillation_new_test_%s.h5"%_suffix)
            
            if tbEpoch%10==0:
                model.save_weights(save_dir + "model_unet3D_distillation_test_%s.h5"%tbEpoch)
                
#             print(val_loss,min_val_loss)


#             if val_loss<min_val_loss:
#                 min_val_loss = val_loss
#                 model.save_weights(save_dir + "model_unet3D.h5")

            del(history)
            gc.collect()
    model.save_weights(save_dir + "model_unet3D_distillation_new_test.h5")

                









if __name__=="__main__":
    data_path = Path('/raid/pengcheng.xu/miccai_endo/')

    train_path = data_path / 'train'

    cropped_train_path = data_path / 'cropped_train'

    original_height, original_width = 1080, 1920
    height, width = 1024, 1280
    h_start, w_start = 28, 320

    binary_factor = 255
    parts_factor = 85
    instrument_factor = 32
    
    train_file_names, val_file_names,test_file_names = get_split(0)
    
    
    train_images = []
    train_masks = []

    for i in range(len(train_file_names)):
        temp_img = load_image(train_file_names[i])
        temp_mask = load_mask(train_file_names[i],'parts')
        train_images.append(temp_img)
        train_masks.append(temp_mask) 
        
        
    val_images = []
    val_masks = []

    for i in range(len(val_file_names)):
        temp_img = load_image(val_file_names[i])
        temp_mask = load_mask(val_file_names[i],'parts')
        val_images.append(temp_img)
        val_masks.append(temp_mask)
        
    train_images_arr = np.array(train_images)
    train_masks_arr = np.array(train_masks)
    
    val_images_arr = np.array(val_images)
    val_masks_arr = np.array(val_masks)

    
    model_dir = "./Unet_parts/"
    
    print("data loaded!")
  
    train_model(total_epochs = 203, save_dir = model_dir,train_images = train_images_arr,train_labels = train_masks_arr,val_images=val_images_arr,val_labels=val_masks_arr)
    print("complete training!")