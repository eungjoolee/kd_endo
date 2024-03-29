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


#loss function for segmentaion
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


# loss function for KD

def st_loss(y_true, y_pred):
    kl_div = tf.keras.losses.KLDivergence()
    temperature = 0.16
    st_loss = kl_div(y_true,y_pred) * temperature * temperature
    
    return st_loss



# here we fixed weight of teacher and self_TA with 1/3, because of the memory issue, we generate features one-by-one
def _generate_feature_by_sub(_img, t_unet,t_self_1_unet,t_self_2_unet):
#     _epoch_ratio_teacher = 1/3
#     epoche_ratio_student = 1/3
#     _sub_index = 1
    _img_feature = np.zeros((_img.shape[0],1024,1280,4))
    
#     _img_feature_teacher = t_unet(_img[:_sub_index])
#     _img_feature_self_1,_ = t_self_1_unet(_img[:_sub_index])
#     _img_feature_self_2,_ = t_self_2_unet(_img[:_sub_index])

    for i in range(_img.shape[0]):
       
        _temp_feature_teacher,_ = t_unet(_img[i:(i+1)])
        _temp_feature_self_1,_ = t_self_1_unet(_img[i:(i+1)])
        _temp_feature_self_2,_ = t_self_2_unet(_img[i:(i+1)])
        _img_feature[i:(i+1)] = (_temp_feature_teacher  + _temp_feature_self_1  + _temp_feature_self_2)/3
        

    return _img_feature





def train_model(total_epochs, save_dir,train_images,train_labels,val_images,val_labels):

    # save log files for disconection of server
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename='train_unet_Gtan_status.log', level=logging.DEBUG, format=LOG_FORMAT)
    
    
    # define student model and GTA, the GTAs have the same setting with student model


    s_model = models.unet_2d((1024, 1280,3), [16, 32, 64, 128], n_labels=4,
                      stack_num_down=1, stack_num_up=1,
                      activation='GELU', output_activation='Softmax', 
                      batch_norm=True, pool='max', unpool='nearest', name='unet')

    t_s_model_1 = models.unet_2d((1024, 1280,3), [16, 32, 64, 128], n_labels=4,
                      stack_num_down=1, stack_num_up=1,
                      activation='GELU', output_activation='Softmax', 
                      batch_norm=True, pool='max', unpool='nearest', name='unet')
    
    t_s_model_2 = models.unet_2d((1024, 1280,3), [16, 32, 64, 128], n_labels=4,
                      stack_num_down=1, stack_num_up=1,
                      activation='GELU', output_activation='Softmax', 
                      batch_norm=True, pool='max', unpool='nearest', name='unet')
    
    
    # define teacher model 

    t_model = models.unet_2d((1024, 1280,3), [64, 128, 256, 512], n_labels=4,
                      stack_num_down=1, stack_num_up=1,
                      activation='GELU', output_activation='Softmax', 
                      batch_norm=True, pool='max', unpool='nearest', name='unet')
    
    # get the soft target of models and build model with output[feature_out,segmenation_out],using the stop_gradient to not training the related model.

    feature_output = tf.stop_gradient(t_model.get_layer('unet_output').output)
    segmentation_output = tf.stop_gradient(t_model.get_layer('unet_output_activation').output)

    t_transunet = Model(inputs = t_model.input, outputs = [feature_output,segmentation_output])
    
    # load the teacher's weight
    
    # t_latest = tf.train.latest_checkpoint("./big_unet/")
    t_transunet.load_weights("./Unet_parts/" + "model_unet3D_distillation_test_40.h5")

    feature_pre = s_model.get_layer('unet_output').output
    segmentation_pre = s_model.get_layer('unet_output_activation').output
    s_unet = Model(inputs = s_model.input, outputs = [feature_pre,segmentation_pre])
    
    
    self_1_feature_output = tf.stop_gradient(t_s_model_1.get_layer('unet_output').output)
    self_1_seg_output = tf.stop_gradient(t_s_model_1.get_layer('unet_output_activation').output)
    self_2_feature_output = tf.stop_gradient(t_s_model_2.get_layer('unet_output').output)
    self_2_seg_output = tf.stop_gradient(t_s_model_2.get_layer('unet_output_activation').output)
    
    t_self_1_unet = Model(inputs = t_s_model_1.input, outputs = [self_1_feature_output,self_1_seg_output])
    t_self_2_unet = Model(inputs = t_s_model_2.input, outputs = [self_2_feature_output,self_2_seg_output])
    
    
    # load the initial pre-trained model weight
    
    # u_latest = tf.train.latest_checkpoint("./small_unet_2/")
    s_unet.load_weights("./Unet_parts_s1/" + "model_unet3D_distillation_test_40.h5")
    t_self_1_unet.load_weights("./Unet_parts_s1/" + "model_unet3D_distillation_test_40.h5")
    t_self_2_unet.load_weights("./Unet_parts_s1/" + "model_unet3D_distillation_test_40.h5")

    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad = True)


    # learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    # learning_rate_boundaries = [125, 250, 500, 240000, 360000]
    # learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    #     boundaries=learning_rate_boundaries, values=learning_rates
    # )

    # optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)

    # combined the loss for training 
    losses_unet = {"unet_output": st_loss, "unet_output_activation": IOU_CCE_loss}
    lossWeights_unet = {"unet_output": 1.0, "unet_output_activation": 1.0}
    s_unet.compile(loss=losses_unet,loss_weights= lossWeights_unet, optimizer=optimizer)


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    nEpochs = total_epochs
    batchSize = 1

    tbEpoch = 0

    num_sample = train_images.shape[0]
    sub_num = 2
    sub_index = [np.int32(num_sample*s/sub_num) for s in range(sub_num+1)  ]

    min_val_loss = 10000.0
    
    aug_img_val, aug_label_val = preprocess_val_data(val_images,val_labels)

            
    for epoch in range(nEpochs):
    #     display.clear_output()
        data_list = np.arange(num_sample)
        np.random.shuffle(data_list)
        if epoch > 1:
                t_self_1_unet.load_weights(save_dir + "model_unet3D_distillation_new_test_0.h5")
                t_self_2_unet.load_weights(save_dir + "model_unet3D_distillation_new_test_1.h5")


        for sub in range(sub_num):  


            tmp_image = train_images[data_list[sub_index[sub]:sub_index[sub+1]],:,:,:].astype(np.float32)
            tmp_label = train_labels[data_list[sub_index[sub]:sub_index[sub+1]],:,:].astype(np.float32)
            tmp_image, tmp_label = preprocess_train_data(tmp_image, tmp_label)

            tmp_feature = _generate_feature_by_sub(tmp_image,t_transunet,t_self_1_unet,t_self_2_unet)

            history = s_unet.fit(tmp_image , [tmp_feature,tmp_label], 
    #                             validation_data=(valid_img, [valid_feature,valid_label]),
                                batch_size=batchSize, 
                                epochs = tbEpoch + 1, 
                                initial_epoch = tbEpoch, 
                                shuffle = False, 
                                verbose = 1)  
            tbEpoch += 1
            
#             _, s_predict = s_unet.predict(aug_img_val,batch_size=1)
#             val_loss = IOU_loss(s_predict,aug_label_val.astype(np.float32))
  
            
            _suffix = tbEpoch % 2
            s_unet.save_weights(save_dir + "model_unet3D_distillation_new_test_%s.h5"%_suffix)
            
            if tbEpoch%5==0:
                s_unet.save_weights(save_dir + "model_unet3D_distillation_test_%s.h5"%tbEpoch)


#             if val_loss<min_val_loss:
#                 min_val_loss = val_loss
#                 s_unet.save_weights(save_dir + "model_unet3D_kd_TA_two.h5")

            del(history)
            gc.collect()


                









if __name__=="__main__":
    data_path = Path('/raid/pengcheng.xu/miccai_endo/')

    # load augmented or not data to train the student
    # load the pretrained weight in line 253

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

    
    model_dir = "./unet_kd_parts/"
    
    print("data loaded!")
  
    train_model(total_epochs = 403, save_dir = model_dir,train_images = train_images_arr,train_labels = train_masks_arr,val_images=val_images_arr,val_labels=val_masks_arr)
    print("complete training!")