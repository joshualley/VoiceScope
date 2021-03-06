import numpy as np
from scipy import misc,ndimage
from scipy.stats import gaussian_kde as kde
from tqdm import *
import cv2

def squre_transform(filename): #读取图像，放大两倍，做平方变换
    pic = misc.imread(filename, flatten = True)
    pic = ndimage.zoom(pic, 2)
    pic = pic**2
    pic = ((pic-pic.min())/(pic.max()-pic.min())*255).round()
    return pic

def decompose(pic): #核密度聚类，给出极大值、极小值点、背景颜色、聚类图层
    print(u'图层聚类分解中...')
    d0 = kde(pic.reshape(-1), bw_method=0.2)(range(256)) #核密度估计
    d = np.diff(d0)
    d1 = np.where((d[:-1]<0)*(d[1:]>0))[0] #极小值
    d1 = [0]+list(d1)+[256]
    d2 = np.where((d[:-1]>0)*(d[1:]<0))[0] #极大值
    if d1[1] < d2[0]:
        d2 = [0]+list(d2)
    if d1[len(d1)-2] > d2[len(d2)-1]:
        d2 = list(d2)+[255]
    dc = sum(map(lambda i: d2[i]*(pic >= d1[i])*(pic < d1[i+1]), range(len(d2))))
    print(u'分解完成. 共%s个图层'%len(d2))
    return dc

def erosion_test(dc): #抗腐蚀能力测试
    print(u'抗腐蚀能力测试中...')
    layers = []
    #bg = np.argmax(np.bincount(dc.reshape(-1)))
    #d = [i for i in np.unique(dc) if i != bg]
    d = np.unique(dc)
    for k in d:
        f = dc==k
        label_im, nb_labels = ndimage.label(f, structure=np.ones((3,3))) #划分连通区域
        ff = ndimage.binary_erosion(f) #腐蚀操作
        def test_one(i):
            index = label_im==i
            if (1.0*ff[index].sum()/f[index].sum() > 0.9) or (1.0*ff[index].sum()/f[index].sum() < 0.1):
                f[index] = False
        ff = map(test_one, trange(1, nb_labels+1))
        layers.append(f)
    print(u'抗腐蚀能力检测完毕.')
    return layers

def pooling(layers): #以模仿池化的形式整合特征
    print('整合分解的特征中...')
    result = sum(layers)
    label_im, nb_labels = ndimage.label(result, structure=np.ones((3,3)))
    def pool_one(i):
        index = label_im==i
        k = np.argmax([1.0*layers[j][index].sum()/result[index].sum() for j in range(len(layers))])
        result[index] = layers[k][index]
    t = map(pool_one, trange(1, nb_labels+1))
    print('特征整合成功.')
    return result

def post_do(pic):
    label_im, nb_labels = ndimage.label(pic, structure=np.ones((3,3)))
    print(u'图像的后期去噪中...')
    def post_do_one(i):
        index = label_im==i
        index2 = ndimage.find_objects(index)[0]
        ss = 1.0 * len(pic.reshape(-1))/len(pic[index2].reshape(-1))**2
        #先判断是否低/高密度区，然后再判断是否孤立区。
        if (index.sum()*ss < 16) or ((1+len(pic[index2].reshape(-1))-index.sum())*ss < 16):
            pic[index] = False
        else:
            a,b,c,d = index2[0].start, index2[0].stop, index2[1].start, index2[1].stop
            index3 = (slice(max(0, 2*a-b),min(pic.shape[0], 2*b-a)), slice(max(0, 2*c-d),min(pic.shape[1], 2*d-c)))
            if (pic[index3].sum() == index.sum()) and (1.0*index.sum()/(b-a)/(d-c) > 0.75):
                pic[index2] = False
    t = map(post_do_one, trange(1, nb_labels+1))
    print('后期去噪完成.')
    return pic

def areas(pic): #圈出候选区域
    print('正在生成候选区域...')
    pic_ = pic.copy()
    label_im, nb_labels = ndimage.label(pic_, structure=np.ones((3,3)))
    def areas_one(i):
        index = label_im==i
        index2 = ndimage.find_objects(index)[0]
        pic_[index2] = True
    t = map(areas_one, trange(1, nb_labels+1))
    return pic_

path = '../test_pic/wz6.png'
p = squre_transform(path)
p = decompose(p)
print(p.shape)
#p = erosion_test(p)
#p = pooling(p)
#p = post_do(p)


cv2.imshow('g', p)
cv2.waitKey(0)