import keras
from configuration.config import PATH
model_name = 'model1/model2.h5'
model = keras.models.load_model(PATH.MODEL_DIR+model_name)

import cv2
import numpy as np
import utils.utils as tool
im = cv2.imread('test_pic/13.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = im.reshape([1,48,48,1])
r = model.predict(im)
r = np.argmax(r)
cls = tool.load_words()
print(cls[r])