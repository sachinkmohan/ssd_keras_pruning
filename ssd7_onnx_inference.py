
import onnxruntime as rt
import time
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
import numpy as np

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast


img_height = 300 # Height of the input images
img_width = 480 # Width of the input images
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

sess = rt.InferenceSession("./ssd7_30_ep_op13.onnx", providers=['CUDAExecutionProvider'])

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

def inference_single_image():
    im2 = cv2.imread('./1478899159823020309.jpg')

    #Converting it into batch dimensions
    resized = cv2.resize(im2, (480,300))
    #print(resized.shape)
    frame2 = np.array(np.expand_dims(resized, axis=0), dtype=np.float32)
    #Detections which returns a list
    detections = sess.run([label_name], {input_name: frame2})
    #List converted to the numpy array
    arr = np.asarray(detections)
    y = np.squeeze(arr, axis=0)

    y_pred_decoded = decode_detections(y,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.45,
                                       top_k=200,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)

    for box in y_pred_decoded[0]:
        xmin = box[-4]
        ymin = box[-3]
        xmax = box[-2]
        ymax = box[-1]
        # print(xmin,ymin,xmax,ymax)
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        # cv2.rectangle(im2, (xmin,ymin),(xmax,ymax), color=color, thickness=2 )
        cv2.rectangle(resized, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 255, 0), thickness=2)
        cv2.putText(resized, label, (int(xmin), int(ymin)), font, fontScale, color, thickness)
    cv2.imshow('detected', resized)
    cv2.waitKey(0)

def inference_video():
    #Reading a dummy image

    cap = cv2.VideoCapture('/home/mohan/git/backups/drive_1_min_more_cars.mp4')
    #cap = cv2.VideoCapture('/home/mohan/git/backups/drive.mp4')
    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        new_frame_time = time.time()
        ret, frame = cap.read()
        resized = cv2.resize(frame, (480, 300))

        frame2 = np.array(np.expand_dims(resized, axis=0), dtype=np.float32)
        # Detections which returns a list
        detections = sess.run([label_name], {input_name: frame2})
        # List converted to the numpy array
        arr = np.asarray(detections)
        y = np.squeeze(arr, axis=0)

        # 4: Decode the raw prediction `y_pred`

        y_pred_decoded = decode_detections(y,
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
        cv2.imshow('im', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def main():
  #inference_single_image()
  inference_video()

if __name__ == "__main__":
    main()