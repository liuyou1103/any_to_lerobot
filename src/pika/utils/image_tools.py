import imageio.v3 as imageio
import numpy as np


# def load_image(image_path):
#     if isinstance(image_path, np.ndarray):
#         return image_path
#     return imageio.imread(image_path)
 
 

def load_image(image_path):
    if isinstance(image_path, np.ndarray):
        img = image_path
    else:
        img = imageio.imread(image_path)
    # 如果是灰度图像 (H, W)，扩展成 (H, W, 1)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)  # 在最后增加一个维度
    return img
