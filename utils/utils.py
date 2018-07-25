from configuration.config import PATH
import os

def load_words():
    """
    加载存储在csv文件中的3557个汉字
    :return: 返回汉字字符串
    """
    ch_path = PATH.CHINESE_CSV
    with open(ch_path, 'rb') as f:
        line = f.readline()
        line = line.decode().strip('\n\r').split(',')
        words = ''
        for one in line:
            words += one
    return words

def read_fontnames(path):
    """
    读取给定路径下的字体文件
    :param path: 存放字体文件的文件夹
    :return: 字体文件名列表
    """
    fn = ''
    for root,dir,filenames in os.walk(path):
        fn = filenames
        pass
    return fn


