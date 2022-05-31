import cv2
import numpy as np
import tensorflow as tf
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from PIL import Image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from easydict import EasyDict as edict

import time

class FrameDetector:

    def __init__(self, display=False):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.FLAG = edict()
        self.FLAG.size = 416
        self.FLAG.tiny = False
        self.FLAG.model = "yolov4"
        self.FLAG.weights = './model/modelYolov4'
        #self.FLAG.weights = './model/yolo_model'
        self.FLAG.framework = "tf"
        self.FLAG.display = display

    def setYoloTensor(self):
        # Loads Yolo model and config. Is called once at the begging of the detection

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(self.FLAG)

        start_time = time.time()
        print("Loading started")

        self.saved_model_loaded = tf.saved_model.load(self.FLAG.weights, tags=[tag_constants.SERVING])
        self.infer = self.saved_model_loaded.signatures['serving_default']

        end_time = time.time()-start_time
        print("Model loaded in: ", end_time)

    def detectYoloTensor(self, frame):
        # License plate detection with YOLO network model. Take whole image as an input
        # and return boxes coordinates with license plates

        if frame is None:
            print("No frame loaded")
        else:
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frameRGB)

        self.FrameNumber += 1

        image_data = cv2.resize(frameRGB, (self.FLAG.size, self.FLAG.size))
        image_data = image_data/255
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        batch_data = tf.constant(image_data)
        pred_bbox = self.infer(batch_data)        # all boxes found by net
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # combined boxes based on IOU
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.25
            )

        # convert tensors to numpy arrays
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        image = utils.draw_bbox(frame, pred_bbox)
        image = Image.fromarray(image.astype(np.uint8))
        result = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        rects = self.__cutBox(frame, pred_bbox, False)

        # result is a frame with boxes
        # rects are coordinates of detected boxes
        return result, rects

    def __cutBox(self, frame, predbox, save=False):
        ## Crop detected plate from orginal image. Return boxes coordinates of area with license plate ##

        frame_w, frame_h = frame.shape[:2]
        boxes, scores, classes, num_boxes = predbox
        border = 5

        rects = []
        for i in range(num_boxes[0]):
            coor = boxes[0][i]
            coor[0] = int(coor[0])
            coor[2] = int(coor[2])
            coor[1] = int(coor[1])
            coor[3] = int(coor[3])

            if coor[0] - border < 0 or coor[0] + border > frame_h:
                continue
            if coor[2] - border < 0 or coor[2] + border > frame_h:
                continue
            if coor[1] - border < 0 or coor[2] + border > frame_w:
                continue
            if coor[2] - border < 0 or coor[2] + border > frame_w:
                continue

            # Box structure: (Xstart, Ystart, Xend, Yend)
            rects.append((int(coor[1])-border, int(coor[0])-border, int(coor[3])+border, int(coor[2])+border))

        # Saving plates image, if specified
        if save:
            path = './DataSet/Plates/'

            for i, box in enumerate(rects):
                plate = frame[box[1]:box[3], box[0]:box[2]]
                cv2.imwrite(path + str(self.FrameNumber) + '_' + str(i) + '.jpg', plate)

        return rects
