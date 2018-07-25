import time
import numpy as np
import tensorflow as tf
import os
from configuration.config import PATH
from utils.utils import read_fontnames, load_words

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def read_from_npy(batch_size, num=1):
    path = PATH.DATASET_DIR+'train_one.npy'
    features = np.load(path).astype(np.float32)
    labels = np.array(list(range(3557))*num)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator

def read_record(path, epochs, batch_size):

    def paser(record):
        features = tf.parse_single_example(
            record,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string),
            }
        )
        label = tf.cast(features['label'], tf.float32)
        image = tf.decode_raw(features['img_raw'], tf.float32)

        return image, label


    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(paser)
    dataset = dataset.shuffle(1000).repeat(epochs).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()

    return iterator

def W(shape):
    init = tf.truncated_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(init)

def B(shape):
    init = tf.constant(0.1)
    return tf.Variable(init)

def fc_layer(x, ws, bs):
    w = W(ws)
    b = B(bs)
    net = tf.matmul(x, w) + b
    return tf.nn.relu(net)

def conv2d_layer(x, ws, bs):
    w = W(ws)
    b = B(bs)
    net = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
    net = tf.nn.relu(net)
    net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return net

def batch_norm_layer(x):
    fc_mean, fc_var = tf.nn.moments(x, axes=[0])
    scale = tf.Variable(tf.ones(x.shape[-1:]))
    shift = tf.Variable(tf.zeros(x.shape[-1:]))
    x = tf.nn.batch_normalization(x, fc_mean, fc_var, shift, scale, variance_epsilon=0.001)
    return x

def built_net(x):
    f_size = 32
    k_size = 5
    x = tf.reshape(x, (-1,48,48,1))
    net = conv2d_layer(x, ws=[k_size,k_size,1,f_size], bs=[f_size])
    net = tf.nn.dropout(net, 0.2)
    net = conv2d_layer(net, ws=[k_size,k_size,f_size,f_size], bs=[f_size])
    net = tf.nn.dropout(net, 0.6)
    net = tf.reshape(net,shape=[-1, 12*12*f_size])
    net = fc_layer(net, ws=[12*12*f_size, 1024], bs=[1024])
    net = batch_norm_layer(net)
    net = tf.nn.dropout(net, 0.6)
    out = tf.nn.softmax(fc_layer(net, ws=[1024, 3557], bs=[3557]))
    return out

def optimizer(y_in, y_out):
    y_out = tf.reshape(y_out, (-1, 3557))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_in, logits=y_out))
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_in, 1), tf.argmax(y_out, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return train_op, loss, acc

import cv2
import matplotlib.pyplot as plt
def main():
    words = load_words()
    sess = tf.InteractiveSession()
    path = PATH.DATASET_DIR+'test7x1x2.record'
    #iterator = read_record(path, epochs=10, batch_size=1024)
    iterator = read_from_npy(batch_size=1024)

    x, y = iterator.get_next()
    y = tf.cast(y, tf.int64)
    x = tf.reshape(x, (-1, 48,48,1))
    y_in = tf.one_hot(y, 3557, dtype=tf.int64)
    y_out = built_net(x)
    train_op, loss, acc = optimizer(y_in, y_out)
    tf.global_variables_initializer().run()
    t0 = time.time()
    for epoch in range(1, 100):
        for i in range(235):
            sess.run(train_op)
            if i%10 == 0:

                x_,y_ = sess.run([y,y_out])
                print('in: %s\nout: %s' %(x_[1:10], np.argmax(y_, 1)[1:10]))
                losses, accurary = sess.run([loss, acc])
                print('[==>] Epoch: %d \tStep: %d \tLoss: %s \tAcc: %s \tTime: %ss'
                      %(epoch, i, losses, accurary, round(time.time()-t0, 2)))
                t0 = time.time()


"""
for k in range(len(x_)):
    cv2.imshow('i', np.asarray(x_[k]))
    cv2.waitKey(0)
    print('key:',words[y_[k]])
    sums = []
    for m in range(48):
        s = 0
        for n in range(48):
            s += x_[k][m,n]
            sums.append(s)
    plt.plot(sums)
    plt.show()
"""


if __name__ == "__main__":
    main()
