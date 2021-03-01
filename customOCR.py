import tensorflow as tf
import numpy as np
import string
import copy
from cv2 import resize


class CustomOCR:
    def __init__(self):
        model = tf.keras.models.load_model('.\\model\\CustomOCR')
        self.probModel = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

        alphabet = list(string.ascii_uppercase)
        numbers = [str(k) for k in range(10)]
        self.classes = numbers+alphabet
        self.inputSize = (30, 50)
        self.thresh = 0.4

    def prediction(self, char):
        img = copy.deepcopy(char)

        img = resize(img, self.inputSize)
        img = np.expand_dims(img, 0).astype(np.float32)/255

        result = self.probModel.predict(img)
        result_index = np.argmax(result)
        resultLabel = self.classes[result_index]

        if result[0][result_index] > self.thresh:
            return resultLabel
        else:
            return ""