from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil
import cv2
from skimage import measure
from tqdm import tqdm
from PIL import Image
import math
import imageio
import torch
from astropy.modeling import models, fitting


def recreate_dir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)


def splitImg(im, filename, split_size, shift_size=None):
    split_img_list = []
    split_filename_list = []

    if shift_size is None:
        w, h = im.shape
        patch_x_num = math.ceil(w / split_size)
        patch_y_num = math.ceil(h / split_size)

        for i in range(patch_x_num):
            for j in range(patch_y_num):
                begin_x = i * split_size
                end_x = (i + 1) * split_size
                begin_y = j * split_size
                end_y = (j + 1) * split_size

                patch_im = im[begin_y:end_y, begin_x:end_x]
                patch_filename = '{0}-{1}-{2}-{3}.png'.format( os.path.splitext(filename)[0], str(begin_x), str(begin_y), str(split_size))

                split_img_list.append(patch_im)
                split_filename_list.append(patch_filename)

    else:
        print('to complete split method with shift-size')

    return split_img_list, split_filename_list


def gaussian2D(shape, cen, sigma=1):
    cen_x, cen_y = cen
    m, n = [(ss - 1.) / 2. for ss in shape]
    bias_x = cen_x - n
    bias_y = cen_y - m
    y, x = np.ogrid[-m:m+1,-n:n+1]
    xx = x - bias_x
    yy = y - bias_y

    h = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))

    # x_data = h[int(h.shape[1]/2), :].squeeze()
    # g_init = models.Gaussian1D(amplitude=1., mean=x_data.mean(), stddev=x_data.std())
    # fit_g = fitting.LevMarLSQFitter()
    # ax = np.linspace(0, 2*m, int(2*m+1))
    # g = fit_g(g_init, np.asarray(ax, dtype='float32'), np.asarray(x_data, dtype='float32'))
    # print(g)

    # h[h < np.finfo(h.dtype).eps * h.max()] = 0
    r_thres = m
    r2 = xx * xx + yy * yy
    circmask = r2 > (r_thres * r_thres)
    h[circmask] = 0
    h = h / h.max()
    # h[h>thres] = 1
    return h


def genMask_circle(labelpath, softCircle=False):
    h = 8192
    w = 8192
    maskimg = np.asanyarray(np.zeros([h, w]), dtype='uint8')

    df = pd.read_table(labelpath, sep='\s+', header=None, encoding='utf-8',
                       names=['col', 'row', 'pixnum', 'length', 'width', 'graysum', 'bgavg', 'bgvar'])

    # max_light = np.max(df.values[:, 5] / df.values[:, 2])
    y, x = np.ogrid[:h, :w]
    min_prob = 1
    for _, row in df.iterrows():
        center_x = float(row['col'])
        center_y = float(row['row'])
        width = float(row['length'])
        length = float(row['width'])

        # import pdb;
        # pdb.set_trace()
        # draw gaussian
        begin_x = max(0, int(round(center_x - length)))
        end_x = min(int(begin_x + 2 * length)+1, w)
        begin_y = max(0, int(round(center_y - width)))
        end_y = min(int(begin_y + 2 * width)+1, h)

        line_x = x[:, begin_x:end_x]
        line_y = y[begin_y:end_y, :]

        # gen circle
        radius = width *0.5
        r2 = (line_x - center_x) ** 2 + (line_y - center_y) **2
        circmask = r2 < (radius ** 2)
        try:
            if np.sum(maskimg[begin_y:end_y, begin_x:end_x][circmask] != 0) != 0:
                circmask = r2 < (radius - 0.5) ** 2

            if not softCircle:
                maskimg[begin_y:end_y, begin_x:end_x][circmask] = circmask[circmask] * 255
            else:
                sigma = radius * 2 / 2.355
                gs = np.exp(-r2 / (2 * sigma ** 2))
                gs = gs / gs.max()
                # import pdb;
                # pdb.set_trace()
                circmask = circmask > 0.5
                maskimg[begin_y:end_y, begin_x:end_x][circmask] = np.asarray(gs[circmask] * 255, dtype='uint8')
                if gs[circmask].min() < min_prob:
                    min_prob = gs[circmask].min()
        except Exception:
            print()
    # print(min_prob)
    return maskimg


if __name__ == '__main__':
    base_dir = 'D:/DNRFA-Net/dataset/fitstrans/fitsdata/8192'
    mask_out_dir = 'D:/DNRFA-Net/dataset/fits_softCircle_modified/8192_masks/'
    fits_out_dir = 'D:/DNRFA-Net/dataset/fits_softCircle_modified/8192_fits/'

    split_size = 256

    recreate_dir(mask_out_dir)
    
    recreate_dir(fits_out_dir)

    for filename in tqdm(os.listdir(base_dir)):

        if filename[-4:] == '.fit' or filename[-4:] == 'fits':
            hdul = fits.open(os.path.join(base_dir, filename))
            imgarray = np.array(hdul[0].data)
            split_img_list, split_filename_list = splitImg(imgarray, filename, split_size)
            
            for im, fname in zip(split_img_list, split_filename_list):      # save fits
                new_hdul = fits.HDUList([fits.PrimaryHDU(im)])
                new_hdul.writeto(os.path.join(fits_out_dir, fname.replace(".png", ".fits")), overwrite=True)

        if filename[-4:] == '.IPD':
            maskarray = genMask_circle(os.path.join(base_dir, filename), softCircle=True)  # generate whole-image mask
            split_img_list, split_filename_list = splitImg(maskarray, filename, split_size)
            for im, fname in zip(split_img_list, split_filename_list):      # save 8bit binary mask
                mask = Image.fromarray(im)
                mask.convert('L').save(os.path.join(mask_out_dir,  os.path.splitext(fname)[0]+'.png'))

    print("done")

