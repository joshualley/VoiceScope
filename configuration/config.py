import os

class CMD():
    """语音模块相应指令
    """
    LIGHT_TURN_ON = b'\x01'
    LIGHT_TURN_OFF = b'\x02'
    WINDOWS_MAGNIFY = b'\x05'
    WINDOWS_SHRINK = b'\x06'
    MODE_ONE = b'\x03'
    MODE_TWO = b'\x04'
    TAKE_POTHO = b'\x07'

def get_root():
    config_dir = os.getcwd()
    s = config_dir.split('/')
    prefix = ''
    for i in range(len(s)-1):
        prefix += s[i]+'/'
    return prefix


class PATH():
    """
    项目文件相关路径
    """
    ROOT_DIR = get_root()
    MODEL_DIR = ROOT_DIR+'model/'
    DATASET_DIR = ROOT_DIR+'data/datasets/'
    FONT_DIR = ROOT_DIR+'data/font/'
    LOG_DIR = ROOT_DIR+'data/log/'
    FONT_DISPLAY = FONT_DIR+'display/msyh.ttc'
    TEST_FONT_DIR = FONT_DIR+'test/'
    TRAIN_FONT_DIR = FONT_DIR+'train/'
    CHINESE_CSV = ROOT_DIR+'data/chinese/ch.csv'







