import keras as K
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from configuration.config import CMD, PATH
from utils.utils import load_words, read_fontnames
import cv2
from matplotlib import pyplot as plt
import os
import time

def gene_data(words, fonts=['Dengl.ttf'], font_path = '../data/font/train/'):
    data = []
    for i,word in enumerate(words):
        #print('Index: %d, Word: %s' %(i, word))
        for fontname in fonts:
            for size in range(40,42,2):
                font = ImageFont.truetype(font=font_path+fontname, size=size)
                fsize = font.getsize(text=word)
                image = Image.new('L',(48,48),0)
                draw = ImageDraw.Draw(image)
                #pos_c = (48-fsize[0])/2, (48-fsize[1])/2
                pos_c = (0,0)
                draw.text(pos_c, text=word, font=font,fill=(255))
                image = np.array(list(image.getdata())).reshape(48,48,1)
                data.append(image)
    return data

def record_node(sums):
    # 记录需要切割点的位置
    p_f = []  # 文本行上边缘或文字前边缘位置
    p_b = []  # 文本行下边缘或文字后边缘位置
    pos = []
    for i, pixel_valve in enumerate(sums):
        if i == 0:
            if pixel_valve != 0:
                p_f.append(i)
                pos.append(i)
            continue
        if i == len(sums) - 1:
            if pixel_valve != 0:
                p_b.append(i)
                pos.append(i)
        if sums[i - 1] < 5 and pixel_valve > 5:
            p_f.append(i)
            pos.append(i)
        if sums[i - 1] > 5 and pixel_valve < 5:
            p_b.append(i)
            pos.append(i)

    while len(p_f) != len(p_b):
        print('p_f:',p_f)
        print('p_b:', p_b)
        if len(p_f) < len(p_b):
            p_b.pop(-1)
        if len(p_f) > len(p_b):
            p_f.pop(-1)
    avg = 0
    num = len(p_f)
    for i in range(num):
        avg += (p_b[i] - p_f[i]) / num
    return p_f, p_b, avg

def findRowRegions(im):
    #统计每一行的像素
    rows = []
    for row in im:
        col_sum = 0
        for col in row:
            col_sum += col / len(row)
        rows.append(col_sum)

    p_f, p_b, avg_height = record_node(rows)
    # print(p_f,'\n',p_b,'\n',avg_height)
    row_regions = []
    for i in range(len(p_f)):
        if p_b[i]-p_f[i] < 5:
            continue
        region = im[p_f[i]:p_b[i], :].copy()
        #cv2.imshow('h', region)
        #cv2.waitKey(0)
        row_regions.append(region)
        plt.imshow(region)
        plt.draw()
        plt.pause(0.0001)

    return row_regions

def findColumnRegions(im):
    h, w = im.shape
    scale = h / 40
    im = cv2.resize(im, (int(w / scale), int(40)), interpolation=cv2.INTER_CUBIC)
    # 统计图片每一列的像素平均值
    cols = []
    for col in im.T:
        row_sum = 0
        for row in col:
            row_sum += row / len(col)
        cols.append(row_sum)

    p_f, p_b, avg_width = record_node(cols)
    #print(avg_width)
    # 生成分割后的图片区域
    regions = []
    for i in range(len(p_f)):
        if p_b[i] - p_f[i] < avg_width-1:
            continue
        region = im[:, p_f[i]:p_b[i]].copy()
        if avg_width <= 20 and avg_width >= 10:
            region = cv2.resize(region, (int(avg_width) * 2, int(avg_width) * 2), cv2.INTER_CUBIC)
            #print('shape:', region.shape)
        elif avg_width < 10:
            continue
            #region = cv2.resize(region, (int(avg_width) * 4, int(avg_width) * 4), cv2.INTER_CUBIC)
            #print('shape:',region.shape)
        regions.append(region)

    return regions

def shape_nom(regions):
    reshape_regions = []
    for region in regions:
        h, w = region.shape
        if h > 48:
            region = cv2.resize(region, (48, w), cv2.INTER_CUBIC)
        elif w > 48:
            region = cv2.resize(region, (h, 48), cv2.INTER_CUBIC)
        h, w = region.shape
        h_pad = (0, 48 - h)
        w_pad = (0, 48 - w)
        region = np.pad(region, (h_pad, w_pad), 'constant', constant_values=(0, 0))
        region = region.reshape(48, 48, 1)
        reshape_regions.append(region)

    return reshape_regions

def gene_region(path):
    img = cv2.imread(path)
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #kernel = np.ones((2, 2),np.uint8)
    #im = cv2.erode(im, kernel)
    #im = cv2.dilate(im, kernel)
    res, im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
    im = cv2.bitwise_not(im)
    plt.ion()
    plt.figure()
    plt.show()
    row_regions = findRowRegions(im)
    print('文本行数:', len(row_regions))
    regions = []
    for i, col_im in enumerate(row_regions):
        region = findColumnRegions(col_im)
        regions.extend(region)

        print('%d行文字个数： %d'%(i, len(regions)))
    new_regions = []
    for region in regions:
        region = findColumnRegions(region)
        if len(region) != 1:
            print(len(region))
        new_regions.extend(region)
    resahpe_regions = shape_nom(new_regions)

    return resahpe_regions

def predict(data, model, save=True, cls=None):
    im = np.round(data/255.)
    if cls == None:
        cls = load_words()
    st = time.time()
    predicts = model.predict(im)
    result = ''
    for i,probabilities in enumerate(predicts):
        index = np.argmax(probabilities)
        if probabilities[index] < 0.01:
            continue
        result += cls[index]
        if save:
            nrootdir = ("./pic/")
            if not os.path.isdir(nrootdir):
                os.makedirs(nrootdir)
            cv2.imwrite(nrootdir + cls[index] + ".jpg", data[i])
    print('Result: %s' % (result))
    et = time.time()
    print('花费时间: %s' % (et - st))
    return result

def recognize_aticle(filename):
    datas = gene_region(filename)
    data = np.array(datas).reshape(len(datas), 48, 48, 1)
    model_path = PATH.MODEL_DIR + 'model28x3_1.h5'
    model = K.models.load_model(model_path)
    predict(data, model)

def main():
    cls = load_words()
    model_path = PATH.MODEL_DIR + 'model28x3_1.h5'
    print('Loading model...')
    model = K.models.load_model(model_path)

    train_font_dir = PATH.TRAIN_FONT_DIR
    fonts = read_fontnames(train_font_dir)

    while True:
        words = input('请输入要生成的图片文字\n')
        data = gene_data(words, [fonts[3]], train_font_dir)
        images = []
        for image in np.array(data):
            kernel = np.ones((1, 1), np.uint8)
            erosion = cv2.erode(image, kernel)
            kernel = np.ones((2, 2), np.uint8)
            dilation = cv2.dilate(erosion, kernel)
            dilation = np.round(dilation, 0)
            cv2.imshow('hh', dilation)
            cv2.waitKey(0)
            images.append(dilation)
        images = np.array(images).reshape(-1,48,48,1)
        predict(images, model, cls=cls)

if __name__ == "__main__":
    recognize_aticle('test_pic/wz22.jpg')