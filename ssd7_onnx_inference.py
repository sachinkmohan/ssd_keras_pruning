
import onnxruntime as rt
import time
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
import numpy as np

sess = rt.InferenceSession("./ssd7_30_ep_op13.onnx", providers=['CUDAExecutionProvider'])

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

im2 = cv2.imread('./1478899159823020309.jpg')

#Converting it into batch dimensions
resized = cv2.resize(im2, (480,300))
#print(resized.shape)
frame2 = np.array(np.expand_dims(resized, axis=0), dtype=np.float32)
#Detections which returns a list
detections = sess.run([label_name], {input_name: frame2})
#List converted to the numpy array
arr = np.asarray(detections)
