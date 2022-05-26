'''
Original Code - Pieriluigi ferrari
Modified - Sachin K Mohan <sachinkm308@gmail.com>
Date - 15th May 2022
Version 1
'''

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd7 import build_model
#from models.keras_ssd7_quantize import build_model_quantize
#from models.keras_ssd7_quantize2 import build_model_quantize2
#from keras_loss_function.keras_ssd_loss import SSDLoss  #commented to test TF2.0
from keras_loss_function.keras_ssd_loss_tf2 import SSDLoss # added for TF2.0

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
#from keras_layers.keras_layer_AnchorBoxes_1 import DefaultDenseQuantizeConfig
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

## imports used for pruning
#import tensorflow_model_optimization as tfmot

import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

import time
#prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude


'''
physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print(physical_devices)
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
'''
#from tensorflow.python.keras.backend.tensorflow_backend import set_session
from tensorflow.python.keras.backend import set_session

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.004
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# ## 1. Set the model configuration parameters

img_height = 300 # Height of the input images
img_width = 480 # Width of the input images
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 5 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size

classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist',
           'light']  # Just so we can print class names onto the image instead of IDs
font = cv2.FONT_HERSHEY_SIMPLEX

# fontScale
fontScale = 0.5

# Blue color in BGR
color = (255, 255, 0)

# Line thickness of 2 px
thickness = 1
model_path = './saved_models/ssd7_base_epoch-30_loss-2.0457_val_loss-2.2370.h5'

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

#K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'compute_loss': ssd_loss.compute_loss})


def inference_single_image():
    #Reading a dummy image
    im2 = cv2.imread('./1478899365487445082.jpg')
    #im2 = image.img_to_array(im2)

    #Converting it into batch dimensions
    im3 = np.expand_dims(im2, axis=0)
    #print(im3.shape)


    # Make a prediction

    y_pred = model.predict(im3)

    #np.save('array_ssd7_pc.npy', y_pred)
    # 4: Decode the raw prediction `y_pred`

    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.45,
                                       top_k=200,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    #print(y_pred_decoded[0])
    #print(y_pred_decoded)
    #print(len(y_pred_decoded))


    ## Drawing a bounding box around the predictions


    for box in y_pred_decoded[0]:
        xmin = box[-4]
        ymin = box[-3]
        xmax = box[-2]
        ymax = box[-1]
        #print(xmin,ymin,xmax,ymax)
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        #cv2.rectangle(im2, (xmin,ymin),(xmax,ymax), color=color, thickness=2 )
        cv2.rectangle(im2, (int(xmin),int(ymin)),(int(xmax),int(ymax)), color=(0,255,0), thickness=2 )
        cv2.putText(im2, label, (int(xmin), int(ymin)), font, fontScale, color, thickness)


    # In[ ]:

    '''
    value = True
    while (value):
        cv2.imshow('frame', im2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        value = False
    '''
    cv2.imshow('frame', im2)
    cv2.waitKey(0)

    #cv2.destroyAllWindows()

def inference_video():
    #Reading a dummy image

    cap = cv2.VideoCapture('/home/agrosy/git/drive.mp4')
    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        new_frame_time = time.time()
        ret, frame = cap.read()
        resized = cv2.resize(frame, (480, 300))
        #im2 = image.img_to_array(im2)

        #Converting it into batch dimensions
        im3 = np.expand_dims(resized, axis=0)

        # Make a prediction
        y_pred = model.predict(im3)

        # 4: Decode the raw prediction `y_pred`

        y_pred_decoded = decode_detections(y_pred,
                                           confidence_thresh=0.5,
                                           iou_threshold=0.45,
                                           top_k=200,
                                           normalize_coords=normalize_coords,
                                           img_height=img_height,
                                           img_width=img_width)

        np.set_printoptions(precision=2, suppress=True, linewidth=90)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)

        ## Drawing a bounding box around the predictions
        for box in y_pred_decoded[0]:
            xmin = box[-4]
            ymin = box[-3]
            xmax = box[-2]
            ymax = box[-1]
            #print(xmin,ymin,xmax,ymax)
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            #cv2.rectangle(im2, (xmin,ymin),(xmax,ymax), color=color, thickness=2 )
            cv2.rectangle(resized, (int(xmin),int(ymin)),(int(xmax),int(ymax)), color=(0,255,0), thickness=2 )
            cv2.putText(resized, label, (int(xmin), int(ymin)), font, fontScale, color, thickness)
            cv2.putText(resized, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            print(fps)
        #cv2.imshow('im', resized)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    cv2.destroyAllWindows()
        #    break

def main():
    #inference_single_image() # Uncomment any one of them based on your needs
    inference_video()

if __name__ == "__main__":
    main()

