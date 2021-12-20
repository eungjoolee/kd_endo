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

# for data augmentation
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop
)




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

# split the data for training and validation
def get_split(fold):
    folds = {0: [1, 3],
             1: [2, 5],
             2: [4, 8],
             3: [6, 7]}

    train_path = data_path / 'cropped_train'

    train_file_names = []
    val_file_names = []

    for instrument_id in range(1, 9):
        if instrument_id in folds[fold]:
            val_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))
        else:
            train_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))

    return train_file_names, val_file_names


# augmentation for training dataset and validation dataset
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



def IOU_loss(y_true, y_pred):
    y_pred = K.batch_flatten(y_pred)
    y_true = K.batch_flatten(y_true)
#     y_true = K.batch_flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=2))
    
    class_intersection = K.sum(y_true * y_pred)
    class_loss = (class_intersection + 1) / (K.sum(y_true) + K.sum(y_pred) - class_intersection + 1)
    return 1 - class_loss







# def train_model_kd(total_epochs,save_dir,images,labels):
def train_model(total_epochs, save_dir,train_images,train_labels,val_images,val_labels):


    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename='train_unet_status_big.log', level=logging.DEBUG, format=LOG_FORMAT)


    # initialize the model, if you use multiple classs segmenation you need changes n_labels=classes+1, and output_activation='Softmax', and loss to tf.keras.losses.CategoricalCrossentropy
    model = models.unet_2d((1024, 1280,3), [64, 128, 256, 512, 512], n_labels=1,
                      stack_num_down=1, stack_num_up=1,
                      activation='GELU', output_activation='Sigmoid', 
                      batch_norm=True, pool='max', unpool='nearest', name='unet')

    

    model.compile(loss=IOU_loss, optimizer=tf.keras.optimizers.Adam(lr=0.0001, amsgrad = True))
   

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    nEpochs = total_epochs
    batchSize = 1

    tbEpoch = 0
    
    # split two sub-datasets for training

    num_sample = train_images.shape[0]
    sub_num = 2
    sub_index = [np.int32(num_sample*s/sub_num) for s in range(sub_num+1)  ]

    min_val_loss = 10000.0
    
    # precess the validation dataset
    aug_img_val, aug_label_val = preprocess_val_data(val_images,val_labels)

            
    for epoch in range(nEpochs):
    
        data_list = np.arange(num_sample)
        np.random.shuffle(data_list)
    
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
            
            s_predict = model.predict(aug_img_val.astype(np.float32),batch_size=1)
            val_loss = IOU_loss(s_predict,aug_label_val.astype(np.float32))
  
            

            # save the weight every 30 epochs, and best weight for validation (here we don't have test dataset, so just save,no further operation)
            if tbEpoch%60==0:
                s_unet.save_weights(save_dir + "model_unet3D_distillation_test_%s.h5"%tbEpoch)


            if val_loss<min_val_loss:
                min_val_loss = val_loss
                model.save_weights(save_dir + "model_unet3D.h5")

            del(history)
            gc.collect()
#         s_unet.save_weights(save_dir + "model_unet3D_distillation_new_test_%s.h5"%_suffix)

                









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
    
    train_file_names, val_file_names = get_split(0)
    
    
    train_images = []
    train_masks = []

    for i in range(len(train_file_names)):
        temp_img = load_image(train_file_names[i])
        temp_mask = load_mask(train_file_names[i],'binary')
        train_images.append(temp_img)
        train_masks.append(temp_mask) 
        
        
    val_images = []
    val_masks = []

    for i in range(len(val_file_names)):
        temp_img = load_image(val_file_names[i])
        temp_mask = load_mask(val_file_names[i],'binary')
        val_images.append(temp_img)
        val_masks.append(temp_mask)
        
    train_images_arr = np.array(train_images)
    train_masks_arr = np.array(train_masks)
    
    val_images_arr = np.array(val_images)
    val_masks_arr = np.array(val_masks)

    
    model_dir = "./unet_0_new/"
    
    print("data loaded!")
  
    train_model(total_epochs = 403, save_dir = model_dir,train_images = train_images_arr,train_labels = train_masks_arr,val_images=val_images_arr,val_labels=val_masks_arr)
    print("complete training!")