import keras
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import sys
#sys.path.append('../')
from configuration.config import CMD, PATH
from utils.utils import load_words
import cv2
import time
import os

def findContour(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(grey, kernel)
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(erosion, kernel)
    res, thresh = cv2.threshold(dilation, 0, 255, cv2.THRESH_OTSU)
    color_translated = cv2.bitwise_not(thresh)
    img, contours, hierarchy = cv2.findContours(color_translated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    xy = []
    cv2.imshow('bit', dilation)
    res, grey_thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_OTSU)
    color_translated = cv2.bitwise_not(grey_thresh)
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        if w > 20 and w <= 200 and h > 20 and h <= 200:
            region = color_translated[y + 2:y + h - 2, x + 2:x + w - 2].copy()
            scaleW, scaleH = w / 42, h / 42
            scale = max(scaleH, scaleW)
            region = cv2.resize(region, (min(int(w / scale), 48), min(int(h / scale), 48)))
            w1, h1 = region.shape
            h_pad = (0, 48-h1)
            w_pad = (48-w1, 0)
            region = np.pad(region, (w_pad, h_pad), 'constant', constant_values=(0, 0))
            region = region.reshape(48, 48, 1)
            regions.append(region)
            xy.append((x, y))

    return image, regions, xy

def predict(imgs, model, xy, save=False, cls=None):
    img01 = imgs/255
    #print(imgs)
    if cls == None:
        cls = load_words()
    st = time.time()
    predicts = model.predict(img01)

    result = ''
    new_xy = []
    for i,probabilities in enumerate(predicts):
        index_cls = np.argmax(probabilities)
        if index_cls == 3557:
            print(index_cls)
        if index_cls == 3557 or probabilities[index_cls] < 0.8:
            continue
        new_xy.append(xy[i])
        result += cls[index_cls]
        if save:
            nrootdir = ("./cut_image/")
            if not os.path.isdir(nrootdir):
                os.makedirs(nrootdir)
            cv2.imwrite(nrootdir + cls[index_cls] + ".jpg", imgs[i])
    #print('Result: %s' % (result))
    et = time.time()
    #print('花费时间: %s' % (et - st))
    return result, new_xy

import threading
def draw_frame(serial, frame, regions, xy, model, cls):
    if len(regions) != 0:
        r,xy = predict(regions, model, xy, cls)
        if r != '':
            t = threading.Thread(target=speak_msg, args=(r, serial))
            t.run()
            pass
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for i in range(len(r)):
            font = ImageFont.truetype(font=PATH.FONT_DISPLAY, size=32)
            draw = ImageDraw.Draw(frame)
            draw.text(xy[i], r[i], (0, 0, 255), font)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    return frame


def speak_msg(msg, serial):
    #print('----------------------------------------------')
    msg = '@TextToSpeech#[m3][t10]' + str(msg) + '$'
    msg = bytes(msg.encode('gbk'))
    #print(msg)
    serial.write(msg)

def turn_on():
    pass

def turn_off():
    pass

def controler(state, serial):
    global part_num
    cmd = serial.read_all()
    if cmd == CMD.LIGHT_TURN_ON:
        turn_on()
    elif cmd == CMD.LIGHT_TURN_OFF:
        turn_off()
    elif cmd == CMD.WINDOWS_MAGNIFY:
        #part_num = 1
        state = MAGNIFY
        print(state)
    elif cmd == CMD.WINDOWS_SHRINK:
        part_num = 4
        state = RUN
        print(state)
    elif cmd == CMD.TAKE_POTHO:
        pass
    return state

MAGNIFY = 0
RUN = 1
def action(state, show_frame):
    if state == 0:
        shape = show_frame.shape
        show_frame = cv2.resize(show_frame, (shape[1] * 3, shape[0] * 3))
    return show_frame

part_num = 1
import serial.tools.list_ports
def get_frame_from_camera():

    print('Get serial...')
    p = serial.tools.list_ports.comports()[0]
    serialName = p[0]
    serialFd = serial.Serial(serialName, 9600, timeout=60)

    print('Loading classes...')
    cls = load_words()
    print('Loading model...')
    #model_path = PATH.MODEL_DIR + 'model28x3_1.h5'
    model_path = PATH.MODEL_DIR + 'model_keras/model.h5'
    model = keras.models.load_model(model_path)

    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)

    serialFd.write(b'@PlayFlashText#099$')

    state = RUN
    while True:
        _, frame = camera.read()
        w, h, c = frame.shape
        s = int(part_num/2)
        w, h = int(w/part_num), int(h/part_num)
        focus_frame = frame[w*s:w*(s+1), h*s:h*(s+1), :]

        processed_frame, regions, xy = findContour(focus_frame)
        regions = np.array(regions)
        show_frame = draw_frame(serialFd, processed_frame, regions, xy, model, cls)

        state = controler(state, serialFd)
        show_frame = action(state, show_frame)
        #cv2.rectangle(show_frame,(w*s, h*s),(w*(s+1), h*(s+1)),(255, 0, 0))
        cv2.imshow('processed frame', show_frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


def main():
    get_frame_from_camera()

if __name__ == '__main__':
    main()
