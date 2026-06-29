from astropy.io import fits
import numpy as np
import os
import pandas as pd
import shutil
from tqdm import tqdm
from PIL import Image
from scipy.optimize import curve_fit,minimize

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

def fit_global_theta(image_data, star_positions):
    def global_fit_function(params, xy, data):
        theta = params[0]
        residuals = 0
        for (center_x, center_y, length, width) in star_positions:
            sigma_x = length / 2.355
            sigma_y = width / 2.355
            amplitude = np.max(data)
            offset = np.min(data)
            fitted_data = gaussian_2d_rot(xy, center_x, center_y, sigma_x, sigma_y, amplitude, offset, theta)
            residuals += np.sum((data.ravel() - fitted_data.ravel())**2)
        return residuals

    initial_stars = star_positions[:min(10, len(star_positions))]
    x = np.arange(image_data.shape[1])
    y = np.arange(image_data.shape[0])
    x, y = np.meshgrid(x, y)

    initial_guess = [0]
    bounds = [(-np.pi/2, np.pi/2)]

    result = minimize(global_fit_function, initial_guess, args=((x, y), image_data), bounds=bounds, method='L-BFGS-B')
    global_theta = result.x[0]
    return global_theta

def gaussian_fitting(image_data, center_x, center_y, length, width, global_offset, global_theta):
    x_min = max(0, int(np.floor(center_x -  width)))
    x_max = min(image_data.shape[1], int(np.ceil(center_x +  width)))
    y_min = max(0, int(np.floor(center_y -  length)))
    y_max = min(image_data.shape[0], int(np.ceil(center_y +  length)))

    if x_min >= x_max or y_min >= y_max:
        return np.zeros_like(image_data)

    local_data = image_data[y_min:y_max, x_min:x_max]
    if local_data.size == 0 or np.all(local_data == 0):
        return np.zeros_like(image_data)

    x = np.arange(x_min, x_max)
    y = np.arange(y_min, y_max)
    x, y = np.meshgrid(x, y)

    def local_fit_function(xy, sigma, amplitude, offset):
        return gaussian_2d_rot(xy, center_x, center_y, sigma, sigma * (width / length), amplitude, offset, global_theta).ravel()

    ini_sigma=length / 2.355
    initial_guess = (ini_sigma, np.max(local_data)-global_offset, global_offset)
    bounds = ([0.5*ini_sigma, 0, 0], [1.5*ini_sigma, np.max(local_data) , np.max(local_data)])

    try:
        popt, _ = curve_fit(local_fit_function, (x, y), local_data.ravel(), p0=initial_guess, bounds=bounds, maxfev=10000)
        sigma_fit, amplitude_fit, offset_fit = popt

        x_full, y_full = np.meshgrid(np.arange(image_data.shape[1]), np.arange(image_data.shape[0]))
        fitted_image = gaussian_2d_rot((x_full, y_full), center_x, center_y, 2*sigma_fit, 2*sigma_fit * (width / length), amplitude_fit, offset_fit, global_theta)
        final_fitted_image = (fitted_image - np.min(fitted_image)) / (np.max(fitted_image) - np.min(fitted_image)) * 255
        return final_fitted_image

    except (RuntimeError, ValueError) as e:
        x_full, y_full = np.meshgrid(np.arange(image_data.shape[1]), np.arange(image_data.shape[0]))
        sigma_x_default = length / 2.355
        sigma_y_default = width / 2.355
        amplitude_default = np.max(local_data)
        offset_default = np.median(local_data)
        theta_default = 0
        final_fitted_image = gaussian_2d_rot((x_full, y_full), center_x, center_y, 2*sigma_x_default, 2*sigma_y_default, amplitude_default, offset_default, theta_default)
        return final_fitted_image

if __name__ == '__main__':
    base_dir = 'dataset/new_data_ast/ipd'
    fits_dir = 'dataset/new_data_ast/fits'
    mask_out_dir = 'dataset/new_data_ast/gauss2d_masks_2'

    recreate_dir(mask_out_dir)

    for filename in tqdm(os.listdir(base_dir)):
        if filename.endswith('.IPD'):
            label_path = os.path.join(base_dir, filename)
            fits_filename = os.path.splitext(filename)[0] + '.fits'
            fits_path = os.path.join(fits_dir, fits_filename)

            if os.path.exists(fits_path):
                hdul = fits.open(fits_path)
                imgarray = np.array(hdul[0].data, dtype=np.float64)
                global_offset = np.min(imgarray)

                df = pd.read_table(label_path, sep='\s+', header=None, encoding='utf-8',
                                   names=['col', 'row', 'pixnum', 'length', 'width', 'graysum', 'bgavg', 'bgvar'])

                star_positions = df[['col', 'row', 'length', 'width']].values.tolist()
                global_theta = fit_global_theta(imgarray, star_positions)

                supervision_image = np.zeros_like(imgarray, dtype=np.float64)

                common_ratio = np.median(df['width'] / df['length'])

                for _, row in df.iterrows():
                    center_x = float(row['col'])
                    center_y = float(row['row'])
                    length = float(row['length'])
                    width = common_ratio * length
                    fitted_image = gaussian_fitting(imgarray, center_x, center_y, length, width, global_offset, global_theta)
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
