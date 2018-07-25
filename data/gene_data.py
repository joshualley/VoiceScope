from  configuration.config import CMD, PATH
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os
import cv2
from utils.utils import load_words, read_fontnames


def draw_text(word, font, size):
    """
    绘制相应字号的文字于图片上,图片大小为48*48
    :param word: 需要绘制的文字
    :param font: 文字采用的字体文件的路径
    :param size: 字号
    :return: 已绘制的图片对象
    """
    font = ImageFont.truetype(font=font, size=size)
    #fsize = font.getsize(text=word)
    image = Image.new('L', (48, 48), 0)
    draw = ImageDraw.Draw(image)
    # pos_c = (48 - fsize[0]) / 2, (48 - fsize[1]) / 2

    pos = (0, 0)
    draw.text(pos, text=word, font=font, fill=(255))

    return image

def add_noises(image, ns_num):
    """
    为图片添加黑噪点
    :param image: 灰度图片
    :param ns_num: 噪点数量 
    :return: 带有噪音的图片
    """
    w, h = image.size
    x = [int(np.random.random()*w) for i in range(ns_num)]
    y = [int(np.random.random()*h) for i in range(ns_num)]
    for i in range(ns_num):
        image.putpixel((x[i], y[i]), 0)
    return  image

import tensorflow as tf
def gene_tfrecord():
    """
    生成tfrecord数据集文件
    :return: None
    """
    def tf_writer(writer, index, image):
        img_raw = image.tobytes()
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())

    font_size_range = (38,39,1)
    # 字体路径
    font_dir = PATH.TEST_FONT_DIR

    fonts = read_fontnames(font_dir)
    words = load_words()

    tfrecord_flie = PATH.DATASET_DIR+'test7x1x2.record'
    writer = tf.python_io.TFRecordWriter(tfrecord_flie)

    for i,word in enumerate(words):
        print('No.%d: %s' %(i, word))
        for j, fontname in enumerate(fonts):
            font = font_dir + fontname
            for size in range(font_size_range[0], font_size_range[1], font_size_range[2]):
                image = draw_text(word, font, size)
                image_without_noise = np.array(image.getdata())
                image_with_noise = np.array(add_noises(image, 100).getdata())
                for image in [image_without_noise, image_with_noise]:
                    image = (np.array(image) / 255.0).reshape(48,48)
                    cv2.imshow('threshold', image)
                    cv2.waitKey(1)
                    image = Image.fromarray(image)
                    tf_writer(writer, i, image)

    writer.close()

def gene_data(words, fonts, save_path, font_path, font_size_range, add_noise=False):
    """
    生成numpy数据
    :param words: 
    :param fonts: 字体文件名列表
    :param save_path: 数据集保存路径
    :param font_path: 字体文件路径
    :param font_size_range: 字号范围
    :param add_noise: 是否添加噪音
    :return: None
    """
    data = []
    for k,fontname in enumerate(fonts):
        print('font %s:' %k)
        font = font_path + fontname
        for size in range(font_size_range[0],font_size_range[1],font_size_range[2]):
            print('Font: %s, Size: %d' %(fontname, size))
            for i, word in enumerate(words):
                # print('Index: %d, Word: %s' % (i, word))
                image = draw_text(word, font, size)
                image_without_noise = np.array(list(image.getdata())).reshape(48, 48, 1) / 255.
                images = [image_without_noise]
                if add_noise:
                    image_with_noise = np.array(add_noises(image, 100).getdata()).reshape(48,48,1)/255.
                    images.append(image_with_noise)

                for im in images:
                    cv2.imshow('dilation', im)
                    cv2.waitKey(1)
                    data.append(im)
    data = np.array(data)
    np.save(save_path, data)

def gene_np_datasets():
    words = load_words()
    #train_font_dir = PATH.TRAIN_FONT_DIR
    train_font_dir = 'font/display/'

    fonts = read_fontnames(train_font_dir)
    data_path = PATH.DATASET_DIR+'val_data.npy'
    gene_data(words, fonts, data_path, train_font_dir, font_size_range=(40,41,1), add_noise=False)

def last_cls_data():
    path = '../detect/cut_image/'

    for root,dir,fn in os.walk(path):
        fns = [root + f for f in fn]
    data = []
    for fn in fns:
        im = cv2.imread(fn)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        res, thresh = cv2.threshold(im, 0, 1, cv2.THRESH_OTSU)
        cv2.imshow('im', thresh)
        cv2.waitKey(1)
        data.append(thresh.reshape(48,48,1))
    data = np.array(data)
    name = PATH.DATASET_DIR + '3557cls.npy'
    np.save(name, data)


def main():
    #gene_np_datasets()
    #gene_tfrecord()
    last_cls_data()

if __name__ == "__main__":
    main()