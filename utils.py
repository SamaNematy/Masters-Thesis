import numpy as np
from PIL import Image

import tensorflow.compat.v1 as tf

from keras import backend as K


def bright_mae(y_true, y_pred):
    return K.mean(K.abs(y_pred[:,:,:,:3] - y_true[:,:,:,:3]))

def bright_mse(y_true, y_pred):
    return K.mean((y_pred[:,:,:,:3] - y_true[:,:,:,:3])**2)

def bright_AB(y_true, y_pred):
            return K.abs(K.mean(y_true[:,:,:,:3])-K.mean(y_pred[:,:,:,:3]))

def log10(x):
    numerator = K.log(x)
    denominator = K.log(K.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def bright_psnr(y_true, y_pred):
    mse = K.mean((K.abs(y_pred[:,:,:,:3] - y_true[:,:,:,:3])) ** 2)
    max_num = 1.0
    psnr = 10 * log10(max_num ** 2 / mse)
    return psnr

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) 
    K1 = 0.01
    K2 = 0.03
    L = 1  
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                            (sigma1_sq + sigma2_sq + C2)),
                        (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:    
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                            (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                                    (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def bright_SSIM(y_true, y_pred):
    SSIM_loss = tf_ssim(tf.expand_dims(y_pred[:,:,:,0], -1), tf.expand_dims(y_true[:,:,:,0], -1))+tf_ssim(tf.expand_dims(y_pred[:,:,:,1], -1), tf.expand_dims(y_true[:,:,:,1], -1)) + tf_ssim(tf.expand_dims(y_pred[:,:,:,2], -1), tf.expand_dims(y_true[:,:,:,2], -1))
    return SSIM_loss/3

def bright_MS_SSIM(y_true, y_pred):  
    MS_SSIM_loss = tf_ms_ssim(tf.expand_dims(y_pred[:,:,:,0], -1), tf.expand_dims(y_true[:,:,:,0], -1))+tf_ms_ssim(tf.expand_dims(y_pred[:,:,:,1], -1), tf.expand_dims(y_true[:,:,:,1], -1)) + tf_ms_ssim(tf.expand_dims(y_pred[:,:,:,2], -1), tf.expand_dims(y_true[:,:,:,2], -1))
    return MS_SSIM_loss/3

def psnr_cau(y_true, y_pred):
    mse = np.mean((np.abs(y_pred - y_true)) ** 2)
    max_num = 1.0
    psnr = 10 * np.log10(max_num ** 2 / mse)
    return psnr


def data_augmentation(image, mode):
    if mode == 0:
        return image
    elif mode == 1: 
        return np.flipud(image)
    elif mode == 2:
        return np.rot90(image)
    elif mode == 3: 
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:   
        return np.rot90(image, k=2)
    elif mode == 5: 
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:   
        return np.rot90(image, k=3)
    elif mode == 7:
        image = np.rot90(image, k=3)
        return np.flipud(image)

def load_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0

def save_images(filepath, result_1, result_2 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis = 1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')