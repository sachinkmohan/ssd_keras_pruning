#!/usr/bin/env python
# coding: utf-8

# ### Load tensorRT graph

# In[1]:


import tensorflow as tf
from tensorflow.python.platform import gfile

#import tensorflow.contrib.tensorrt

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

import time

GRAPH_PB_PATH_TRT = './converted_trt_graph/trt_graph_base_for_pc_ep30.pb'
#GRAPH_PB_PATH_B = './output_saved_model_dir/frozen_model_base_30.pb'
#GRAPH_PB_PATH_OD = './frozen_model_od/tf_ssd7_model.pb'


tf_config = tf.ConfigProto()
#tf_config.gpu_options.allow_growth = False
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.004
tf_sess1 = tf.Session(config=tf_config)

#loading the graph for OD
with tf.Session() as sess1:
   print("load graph")
   with gfile.FastGFile(GRAPH_PB_PATH_TRT,'rb') as f:
       graph_def1 = tf.GraphDef()
   graph_def1.ParseFromString(f.read())
   sess1.graph.as_default()
   tf.import_graph_def(graph_def1, name='')
   graph_nodes1=[n for n in graph_def1.node]
   names1 = []
   for t in graph_nodes1:
      names1.append(t.name)
                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                            


# ### Loading the pb graph

# In[ ]:


tf.import_graph_def(graph_def1, name='')


# In[ ]:


tf_input1 = tf_sess1.graph.get_tensor_by_name('input_1:0')
print(tf_input1)
tf_predictions1 = tf_sess1.graph.get_tensor_by_name('predictions/concat:0')
print(tf_predictions1)


# ### Inference on live camera data with pb graph

# In[ ]:


import cv2
import numpy as np
#import matplotlib.pyplot as plt

from tensorflow.python.keras.backend import set_session
graph = tf.get_default_graph()


## Drawing a bounding box around the predictions

classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light'] # Just so we can print class names onto the image instead of IDs
font = cv2.FONT_HERSHEY_SIMPLEX
  

# fontScale
fontScale = 0.5
   
# Blue color in BGR
color = (255, 255, 0)
  
# Line thickness of 2 px
thickness = 1


#Capture the video from the camera

cap = cv2.VideoCapture('/home/mohan/git/backups/drive.mp4')
prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    new_frame_time = time.time()
    ret, frame = cap.read()

    image_resized2 = cv2.resize(frame, (480,300))
    #image_resized3 = cv2.resize(frame, (480, 320))

    if ret:
        t0 = time.time()
        with graph.as_default():
            set_session(sess1)
            inputs1, predictions1 = tf_sess1.run([tf_input1, tf_predictions1], feed_dict={
            tf_input1: image_resized2[None, ...]
        })
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)
        print(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)
        
        y_pred_decoded = decode_detections(predictions1,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.45,
                                   top_k=200,
                                   normalize_coords=True,
                                   img_height=300,
                                   img_width=480)
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        
        for box in y_pred_decoded[0]:
            
            xmin = box[-4]
            ymin = box[-3]
            xmax = box[-2]
            ymax = box[-1]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            #cv2.rectangle(im2, (xmin,ymin),(xmax,ymax), color=color, thickness=2 )
            cv2.rectangle(image_resized2, (int(xmin),int(ymin)),(int(xmax),int(ymax)), color=(0,255,0), thickness=2 )
            cv2.putText(image_resized2, label, (int(xmin), int(ymin)), font, fontScale, color, thickness)
            cv2.putText(image_resized2, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Input Images',image_resized2)
        t1 = time.time()
        #print((float(t1 - t0)))

        #cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    else:
        cap.release()
        break

cap.release()
cv2.destroyAllWindows()

