import shutil

import ultralytics
from ultralytics import YOLO
from PIL import Image
import glob
import os
import numpy as np
import cv2 as cv

ultralytics.checks()

#Bounding Box Merge Algorithm
def merge_bounding_boxes(bounding_boxes,image_width,image_height):
  [x,y] = [image_height,image_width]
  [w,h] = [0,0]
  for i in range(0,bounding_boxes.shape[0]):
    if(bounding_boxes.ndim == 1):
      center_x_1 = int(bounding_boxes[0]*image_width)
      center_y_1 = int(bounding_boxes[1]*image_height)
      width_1 = int(bounding_boxes[2]*image_width)
      height_1 = int(bounding_boxes[3]*image_height)
      x = int(center_x_1 - (width_1/2))
      y = int(center_y_1 - (height_1/2))
      w = int(center_x_1 + (width_1/2))
      h = int(center_y_1 + (height_1/2))
    else:
      center_x_1 = int(bounding_boxes[i,0]*image_width)
      center_y_1 = int(bounding_boxes[i,1]*image_height)
      width_1 = int(bounding_boxes[i,2]*image_width)
      height_1 = int(bounding_boxes[i,3]*image_height)
      vert_x = int(center_x_1 - (width_1/2))
      vert_y = int(center_y_1 - (height_1/2))
      vert_w = int(center_x_1 + (width_1/2))
      vert_h = int(center_y_1 + (height_1/2))
      if(vert_x < x):
        x = vert_x
      if(vert_y < y):
        y = vert_y
      if(vert_w > w):
        w = vert_w
      if(vert_h > h):
        h = vert_h
    return np.array([x,y,w,h])

def compute_iou(box1, box2):
  #DATA: box1 = [L1, T1, R1, B1], box2 = [L2, T2, R2, B2]
  L_inter = max(box1[0], box2[0])
  T_inter = max(box1[1], box2[1])
  R_inter = min(box1[2], box2[2])
  B_inter = min(box1[3], box2[3])
  if(R_inter < L_inter) or (B_inter < T_inter):
    return 0
  A_inter = (R_inter - L_inter) * (B_inter - T_inter)
  A1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
  A2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
  A_union = A1 + A2 - A_inter
  iou = A_inter/A_union
  return iou

if __name__ == '__main__':

    # epochs = 1 #@param {type:"slider", min:100, max:1000, step:10}
    # training_option = "Train new CNN" #@param ["Train new CNN", "Resume an interrupted Training"]
    # weight = 'yolov8m.pt' #@param["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt", "yolov8x"]
    # batch = 32 #@param {type:"slider", min:1, max:64, step:1}
    #
    # model = YOLO(weight)
    # data = "training/training_dataset/data.yaml"
    # project = "training/training_results/"
    # model_folder = "training/training_results/train/weights/last.pt"
    #
    # print(type(model))
    #
    #
    # if training_option == 'Train new CNN':
    #   model.train(data=data, epochs=epochs, imgsz=512, batch=batch, cache=True, patience=50, project=project)
    # else:
    #   model.train(data=data, epochs=epochs, imgsz=512, batch=batch, cache=True, patience=50, project=project, resume=True, model=model_folder)
    #
    # data_test = "test/test_dataset/data.yaml"
    # metrics = model.val(data=data_test)

    # YOLO("training/training_results/train/weights/best.pt").predict("detection/dataset/", imgsz=512, save=False, save_txt=True, save_conf=True)

    # YOLO("training/training_results/train/weights/best.pt").predict("detection/video", imgsz=512, save=False, save_txt=True, save_conf=True, project = "video_results/")


    # Printing result to confront
    # precision_training = '/training/training_results/train/F1_curve.png'
    # precision_test = '/training/training_results/val/F1_curve.png'
    #
    # im1 = Image.open(precision_training)
    # im2 = Image.open(precision_test)
    #
    # print("F1 Training")
    # cv2.imshow(im1)
    # print("F1 Test")
    # cv2.imshow(im2)

    video = '2-iYADzxhFw.mp4'
    path = 'video_results/predict3/labels/'
    video_path = 'detection/video/'
    save_path = 'video_results/'
    frames_path = 'video_results/predict3/frames/'
    video_name = video.replace('.mp4', '')

    print('Reading Video: ' + video_path + video)
    v = cv.VideoCapture(video_path + video)
    frame_count = int(v.get(cv.CAP_PROP_FRAME_COUNT))
    fps = v.get(cv.CAP_PROP_FPS)
    video_w = int(v.get(cv.CAP_PROP_FRAME_WIDTH))
    video_h = int(v.get(cv.CAP_PROP_FRAME_HEIGHT))
    print('Video Info' + '  FPS: ' + str(int(fps)) + ' Width: ' + str(video_w) + ' Height: ' + str(video_h))

    frames = None
    for filename in glob.glob(os.path.join(path, '*.txt')):
      with open(os.path.join(os.getcwd(), filename), 'r') as f:
        bounding_boxes = None
        confidences = None
        # Saves the Frame number in an array
        frame_num = int(os.path.basename(filename).replace('.txt', '').replace(video_name + '_', ''))
        frames = (np.vstack((frames, frame_num)) if (frames is not None) else frame_num)
        for line in f:
          cl, label_x, label_y, label_w, label_h, conf = line.split(' ')
          b = float(conf)
          a = np.array([float(label_x), float(label_y), float(label_w), float(label_h)])
          bounding_boxes = (np.vstack((bounding_boxes, a)) if (bounding_boxes is not None) else a)
          confidences = (np.vstack((confidences, b)) if (confidences is not None) else b)
        conf_max = np.amax(confidences)
        # Get Frames from Video
        v.set(cv.CAP_PROP_POS_FRAMES, frame_num)
        res, image = v.read()
        # cv2_imshow(image)
        image_height = video_h
        image_width = video_w
        [x, y, w, h] = merge_bounding_boxes(bounding_boxes, image_width, image_height)
        cv.rectangle(image, (x, y), (w, h), (0, 0, 255), 4)
        cv.putText(image, 'crosswalk ' + "%.2f" % conf_max, (x, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if image is None:
          continue
        cv.imwrite(frames_path + str(frame_num) + '.jpg', image)
    v.release()
    print("Frame Creation Complete. Run the following code to create video and clear frames from disk.")

    cap = cv.VideoCapture(video_path + video)
    vid_writer = cv.VideoWriter(save_path + video, cv.VideoWriter_fourcc(*'mp4v'), fps, (video_w, video_h))
    frame_num = -1;
    while (cap.isOpened()):
        frame_num += 1
        ret, frame = cap.read()
        if ret == True:
            if (frame_num in frames):
                frame = cv.imread(frames_path + str(frame_num) + '.jpg')
            vid_writer.write(frame)
        else:
            break
    cap.release()
    vid_writer.release()

    shutil.rmtree(frames_path)
    print('Result Saved on ' + save_path + video)

