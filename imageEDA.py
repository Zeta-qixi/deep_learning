import hashlib
import PIL
from PIL import Image
from PIL import ImageStat
import numpy as np
import pandas as pd

def get_hash(image):
    '''
    图片转hash
    '''
    md5 = hashlib.md5()
    md5.update(np.array(image).tobytes())
    return md5.hexdigest()

def get_image_meta(image_id, dataset = 'train'):
    '''
    获取包含RGB的 元数据信息
    '''
    image_src = "" #地址
    img = Image.open(image_src)
    extrema = img.getextrema()
    stat = ImageStat.Stat(img)
    
    meta = {
        'image': image_id,
        'dataset': dataset,
        'hash': get_hash(img),
        'R_min': extrema[0][0],
        'R_max': extrema[0][1],
        'G_min': extrema[1][0],
        'G_max': extrema[1][1],
        'B_min': extrema[2][0],
        'B_max': extrema[2][1],
        'R_avg': stat.mean[0],
        'G_avg': stat.mean[1],
        'B_avg': stat.mean[2],
        'height': img.height,
        'width': img.width,
        'format': img.format,
        'mode': img.mode
    }
    return meta

def image_metadata(df, rootdir, mark='train'):
    img_data = []
    for i, image_id in enumerate(df['image']):
        img_data.append(get_image_meta(image_id, mark))
    
    meta_pd = pd.DataFrame(img_data)
    return meta_pd

