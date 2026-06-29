from astropy.io import fits
import numpy as np
import os
import pandas as pd
import shutil
from tqdm import tqdm
from PIL import Image
from scipy.optimize import curve_fit

def recreate_dir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)

def gaussian_2d_rot(xy, x0, y0, sigma_x, sigma_y, amplitude, offset, theta):
    x, y = xy
    a = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
    c = np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)
    exponent = a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2
    return amplitude * np.exp(-exponent) + offset

def gaussian_fitting(image_data, center_x, center_y, length, width):
    x_min = max(0, int(np.floor(center_x - width)))
    x_max = min(image_data.shape[1], int(np.ceil(center_x + width)))
    y_min = max(0, int(np.floor(center_y - length)))
    y_max = min(image_data.shape[0], int(np.ceil(center_y + length)))

    if x_min >= x_max or y_min >= y_max:
        return np.zeros_like(image_data)

    local_data = image_data[y_min:y_max, x_min:x_max]
    if local_data.size == 0 or np.all(local_data == 0):
        return np.zeros_like(image_data)

    center_value = np.max(local_data)
    center_x_int = int(center_x - x_min)
    center_y_int = int(center_y - y_min)
    local_data[center_y_int, center_x_int] = center_value

    x = np.arange(x_min, x_max)
    y = np.arange(y_min, y_max)
    x, y = np.meshgrid(x, y)

    def local_fit_function(xy, sigma_x, sigma_y, amplitude, offset, theta):
        return gaussian_2d_rot(xy, center_x, center_y, sigma_x, sigma_y, amplitude, offset, theta).ravel()

    initial_guess = (3, 3, np.max(local_data), np.median(local_data), 0)
    bounds = ([0, 0, 0, 0, -np.pi/2], [10, 10, np.max(local_data)*2, np.max(local_data), np.pi/2])

    try:
        popt, _ = curve_fit(local_fit_function, (x, y), local_data.ravel(), p0=initial_guess, bounds=bounds, maxfev=10000)
        sigma_x_fit, sigma_y_fit, amplitude_fit, offset_fit, theta_fit = popt

        x_full, y_full = np.meshgrid(np.arange(image_data.shape[1]), np.arange(image_data.shape[0]))
        fitted_image = gaussian_2d_rot((x_full, y_full), center_x, center_y, 2*sigma_x_fit, 2*sigma_y_fit, amplitude_fit, offset_fit, theta_fit)
        fitted_image = (fitted_image - np.min(fitted_image)) / (np.max(fitted_image) - np.min(fitted_image))*255
        return fitted_image

    except (RuntimeError, ValueError) as e:
        x_full, y_full = np.meshgrid(np.arange(image_data.shape[1]), np.arange(image_data.shape[0]))
        sigma_x_default = length / 2.355
        sigma_y_default = width / 2.355
        amplitude_default = np.max(local_data)
        offset_default = np.median(local_data)
        theta_default = 0
        fitted_image = gaussian_2d_rot((x_full, y_full), center_x, center_y, 2*sigma_x_default, 2*sigma_y_default, amplitude_default, offset_default, theta_default)
        return fitted_image

if __name__ == '__main__':
    base_dir = 'dataset/new_data_ast/ipd'
    fits_dir = 'dataset/new_data_ast/fits'
    mask_out_dir = 'dataset/new_data_ast/gauss2d_masks_*2'

    recreate_dir(mask_out_dir)

    for filename in tqdm(os.listdir(base_dir)):
        if filename[-4:] == '.IPD':
            label_path = os.path.join(base_dir, filename)
            fits_filename = os.path.splitext(filename)[0] + '.fits'
            fits_path = os.path.join(fits_dir, fits_filename)

            if os.path.exists(fits_path):
                hdul = fits.open(fits_path)
                imgarray = np.array(hdul[0].data, dtype=np.float64)

                df = pd.read_table(label_path, sep='\s+', header=None, encoding='utf-8',
                                   names=['col', 'row', 'pixnum', 'length', 'width', 'graysum', 'bgavg', 'bgvar'])

                supervision_image = np.zeros_like(imgarray, dtype=np.float64)

                for _, row in df.iterrows():
                    center_x = float(row['col'])
                    center_y = float(row['row'])
                    length = float(row['length'])
                    width = float(row['width'])
                    fitted_image = gaussian_fitting(imgarray, center_x, center_y, length, width)
                    supervision_image += fitted_image

                max_value = np.max(supervision_image)
                if max_value > 0:
                    supervision_image = (supervision_image / max_value) * 255
                    supervision_image = supervision_image.astype(np.uint8)
                else:
                    supervision_image = np.zeros_like(supervision_image, dtype=np.uint8)
                supervision_image = supervision_image.astype(np.uint8)
                supervision_filename = os.path.join(mask_out_dir, os.path.splitext(filename)[0] + '.png')
                Image.fromarray(supervision_image).save(supervision_filename)

    print("done")
