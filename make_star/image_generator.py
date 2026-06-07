import os
from functools import lru_cache

import pandas as pd
import numpy as np
import cv2
from scipy.spatial import cKDTree
from datetime import datetime


# ==========================================
# 1. 核心渲染与标注函数
# ==========================================
def draw_gaussian_spot(img, cx, cy, brightness, sigma=1.2):
    """Draw one sub-pixel Gaussian star spot into a float image."""
    if sigma <= 0:
        return

    height, width = img.shape
    radius = max(2, int(np.ceil(4.0 * sigma)))
    x_center = int(np.floor(cx))
    y_center = int(np.floor(cy))

    x_min = max(0, x_center - radius)
    x_max = min(width, x_center + radius + 1)
    y_min = max(0, y_center - radius)
    y_max = min(height, y_center + radius + 1)

    if x_min >= x_max or y_min >= y_max:
        return

    x = np.arange(x_min, x_max, dtype=np.float32) - np.float32(cx)
    y = np.arange(y_min, y_max, dtype=np.float32)[:, np.newaxis] - np.float32(cy)
    kernel = brightness * np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    img[y_min:y_max, x_min:x_max] += kernel


def draw_motion_blurred_gaussian_spot(
        img,
        cx,
        cy,
        brightness,
        sigma,
        direction,
        trail_length):
    """Draw one Gaussian star as a one-direction motion blur trail."""
    if trail_length <= 0:
        draw_gaussian_spot(img, cx, cy, brightness, sigma)
        return

    direction = np.asarray(direction, dtype=np.float32)
    direction_norm = np.linalg.norm(direction)
    if direction_norm <= 0:
        draw_gaussian_spot(img, cx, cy, brightness, sigma)
        return
    direction = direction / direction_norm

    sample_count = max(4, int(np.ceil(trail_length / 0.75)))
    offsets = np.linspace(0.0, trail_length, sample_count, dtype=np.float32)
    weights = np.linspace(1.0, 0.25, sample_count, dtype=np.float32)
    weights = weights / weights.sum()

    for offset, weight in zip(offsets, weights):
        x = cx + direction[0] * offset
        y = cy + direction[1] * offset
        draw_gaussian_spot(img, x, y, brightness * float(weight) * 1.25, sigma)


def apply_light_pollution(
        gray_img,
        random_seed=None,
        patch_count=None,
        base_brightness=3.0,
        patch_brightness=(18.0, 34.0),
        add_noise=True,
        noise_mean=1.5,
        noise_std=5.5):
    """Add one or two broad light-pollution glows and full-frame noise."""
    rng = np.random.default_rng(random_seed)
    polluted = gray_img.astype(np.float32)
    height, width = polluted.shape

    mask = np.full((height, width), float(base_brightness), dtype=np.float32)
    if patch_count is None:
        patch_count = int(rng.integers(1, 3))

    x_grid = np.arange(width, dtype=np.float32)
    y_grid = np.arange(height, dtype=np.float32)[:, np.newaxis]
    image_diag = np.hypot(width, height)
    for _ in range(patch_count):
        cx = rng.uniform(0.18, 0.82) * width
        cy = rng.uniform(0.18, 0.82) * height
        falloff_radius = rng.uniform(0.25, 0.42) * image_diag
        value = float(rng.uniform(patch_brightness[0], patch_brightness[1]))
        x = x_grid - np.float32(cx)
        y = y_grid - np.float32(cy)
        radius2 = x * x + y * y
        blob = value / (1.0 + radius2 / (falloff_radius * falloff_radius))
        mask += blob.astype(np.float32)

    smooth_size = max(31, int(max(width, height) * 0.06) | 1)
    mask = cv2.GaussianBlur(mask, (smooth_size, smooth_size), 0)

    if add_noise:
        fine_grain = rng.normal(noise_mean, noise_std, (height, width)).astype(np.float32)
        fine_grain_size = max(5, int(max(width, height) * 0.010) | 1)
        fine_grain = cv2.GaussianBlur(fine_grain, (fine_grain_size, fine_grain_size), 0)
        soft_grain = rng.normal(0.0, noise_std * 0.30, (height, width)).astype(np.float32)
        soft_grain_size = max(11, int(max(width, height) * 0.030) | 1)
        soft_grain = cv2.GaussianBlur(soft_grain, (soft_grain_size, soft_grain_size), 0)
        mask += fine_grain + soft_grain

    return np.clip(polluted + mask, 0, 255).astype(np.uint8)


def _default_gaia_catalogue_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gaia.csv')


@lru_cache(maxsize=4)
def _load_gaia_catalogue(csv_path):
    gaia_columns = {'ra', 'dec', 'phot_g_mean_mag', 'pmra', 'pmdec', 'ref_epoch'}
    df = pd.read_csv(csv_path, usecols=lambda column: column in gaia_columns)
    mags_all = df['phot_g_mean_mag'].to_numpy(dtype=np.float64)
    ra_deg_all = df['ra'].to_numpy(dtype=np.float64)
    dec_deg_all = df['dec'].to_numpy(dtype=np.float64)

    if {'pmra', 'pmdec', 'ref_epoch'}.issubset(df.columns):
        pmra = df['pmra'].to_numpy(dtype=np.float64)
        pmdec = df['pmdec'].to_numpy(dtype=np.float64)
        ref_epoch = df['ref_epoch'].to_numpy(dtype=np.float64)
        target_epoch = float(datetime.utcnow().year)
        cos_dec = np.cos(np.radians(dec_deg_all))
        valid_pm = (
            np.isfinite(pmra) &
            np.isfinite(pmdec) &
            np.isfinite(ref_epoch) &
            (np.abs(cos_dec) > 0.1)
        )
        delta_year = target_epoch - ref_epoch[valid_pm]
        ra_deg_all = ra_deg_all.copy()
        dec_deg_all = dec_deg_all.copy()
        ra_deg_all[valid_pm] = (
            ra_deg_all[valid_pm] +
            (pmra[valid_pm] / 1000.0 / 60.0 / 60.0 / cos_dec[valid_pm]) * delta_year
        ) % 360.0
        dec_deg_all[valid_pm] = dec_deg_all[valid_pm] + (
            pmdec[valid_pm] / 1000.0 / 60.0 / 60.0
        ) * delta_year
        dec_deg_all = np.clip(dec_deg_all, -90.0, 90.0)

    ra_rad_all = np.radians(ra_deg_all)
    dec_rad_all = np.radians(dec_deg_all)

    stars_3d_all = np.vstack((
        np.cos(dec_rad_all) * np.cos(ra_rad_all),
        np.cos(dec_rad_all) * np.sin(ra_rad_all),
        np.sin(dec_rad_all)
    )).T

    return mags_all, stars_3d_all, cKDTree(stars_3d_all)


def _radec_to_unit_vector(ra_deg, dec_deg):
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    return np.array([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad)
    ])


def _normalize_center_radec(ra_deg, dec_deg):
    return ra_deg % 360.0, dec_deg


def _star_display_scales(show_star):
    scales = {
        1: (0.55, 0.70),
        2: (1.00, 1.00),
        3: (1.45, 1.25),
        4: (2.00, 1.55),
    }
    if show_star not in scales:
        raise ValueError('show_star must be 1, 2, 3 or 4')
    return scales[show_star]


def _create_observer_rotation(ra_deg, dec_deg, rotation_deg):
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    z_c = _radec_to_unit_vector(ra_deg, dec_deg)
    x_c = np.array([-np.sin(ra_rad), np.cos(ra_rad), 0.0])
    x_c = x_c / np.linalg.norm(x_c)
    y_c = np.array([
        -np.sin(dec_rad) * np.cos(ra_rad),
        -np.sin(dec_rad) * np.sin(ra_rad),
        np.cos(dec_rad)
    ])
    y_c = y_c / np.linalg.norm(y_c)

    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot = cos_t * x_c + sin_t * y_c
    y_rot = -sin_t * x_c + cos_t * y_c
    return np.vstack((x_rot, y_rot, z_c))


def create_star_image_by_fov(
        ra_in,
        dec_in,
        rotation_deg,
        fov_deg,
        img_width=1920,
        img_height=1080,
        limiting_mag=7.0,
        csv_path=None,
        add_noise=False,
        noise_mean=12,
        noise_std=6,
        return_gray=False,
        star_brightness_scale=1.0,
        star_sigma_scale=1.0,
        show_star=2):
    """Create a Gaia star chart from center RA/Dec, roll angle and horizontal FOV.

    Args:
        ra_in (float): Center right ascension in degrees, in [0, 360].
        dec_in (float): Center declination in degrees, in [-90, 90].
        rotation_deg (float): Image roll angle in degrees, in [0, 360].
            Positive values rotate the
            generated chart counter-clockwise on the image plane.
        fov_deg (float): Horizontal field of view in degrees, in [3, 40].
        img_width (int): Output image width in pixels.
        img_height (int): Output image height in pixels.
        limiting_mag (float): Faintest Gaia G magnitude to draw. Values above 7
            are capped at 7 to keep the image sparse.
        csv_path (str | None): Path to gaia.csv.
        add_noise (bool): Whether to add optional sensor noise.
        noise_mean (float): Gaussian noise mean.
        noise_std (float): Gaussian noise standard deviation.
        return_gray (bool): Return a single-channel image when True.
        star_brightness_scale (float): Scale factor for rendered star brightness.
        star_sigma_scale (float): Scale factor for rendered star spot radius.
        show_star (int): Star display level. 1 is current polluted style, 2 is
            current pure style, 3 and 4 are progressively brighter/larger.

    Returns:
        numpy.ndarray: BGR uint8 image by default, or grayscale uint8 if return_gray.
    """
    ra_in = float(ra_in)
    dec_in = float(dec_in)
    rotation_deg = float(rotation_deg)
    fov_deg = float(fov_deg)

    if not 0.0 <= ra_in <= 360.0:
        raise ValueError('ra_in must be in [0, 360] degrees')
    if not -90.0 <= dec_in <= 90.0:
        raise ValueError('dec_in must be in [-90, 90] degrees')
    if not 0.0 <= rotation_deg <= 360.0:
        raise ValueError('rotation_deg must be in [0, 360] degrees')
    if not 3.0 <= fov_deg <= 40.0:
        raise ValueError('fov_deg must be in [3, 40] degrees')
    if img_width <= 0 or img_height <= 0:
        raise ValueError('img_width and img_height must be positive')

    ra_in, dec_in = _normalize_center_radec(ra_in, dec_in)
    rotation_deg = rotation_deg % 360.0
    draw_limit_mag = min(float(limiting_mag), 7.0)
    show_brightness_scale, show_sigma_scale = _star_display_scales(show_star)

    if csv_path is None:
        csv_path = _default_gaia_catalogue_path()
    csv_path = os.path.abspath(csv_path)
    mags_all, stars_3d_all, tree = _load_gaia_catalogue(csv_path)

    fov_rad = np.radians(fov_deg)
    f_px = (img_width / 2.0) / np.tan(fov_rad / 2.0)
    K = np.array([
        [f_px, 0.0, img_width / 2.0],
        [0.0, f_px, img_height / 2.0],
        [0.0, 0.0, 1.0]
    ])

    diag_half_px = np.hypot(img_width / 2.0, img_height / 2.0)
    diag_fov_rad = 2.0 * np.arctan(diag_half_px / f_px)
    search_radius = 2.0 * np.sin(min(np.pi, diag_fov_rad * 1.08) / 2.0)

    z_c = _radec_to_unit_vector(ra_in, dec_in)
    indices = tree.query_ball_point(z_c, search_radius)
    star_map = np.zeros((img_height, img_width), dtype=np.float32)

    if indices:
        R_OBS_J2K = _create_observer_rotation(ra_in, dec_in, rotation_deg)
        local_stars_j2k = stars_3d_all[indices]
        local_mags = mags_all[indices]

        stars_obs = (R_OBS_J2K @ local_stars_j2k.T).T
        front_mask = stars_obs[:, 2] > 0
        stars_obs = stars_obs[front_mask]
        local_mags = local_mags[front_mask]

        if len(stars_obs) > 0:
            coords_homo = (K @ (stars_obs / stars_obs[:, 2:3]).T).T
            u, v = coords_homo[:, 0], coords_homo[:, 1]
            in_frame = (
                (u >= 0) & (u < img_width) &
                (v >= 0) & (v < img_height) &
                (local_mags <= draw_limit_mag)
            )
            u, v, local_mags = u[in_frame], v[in_frame], local_mags[in_frame]

            for x, y, mag in zip(u, v, local_mags):
                mag_delta = max(0.0, draw_limit_mag - mag)
                sigma = star_sigma_scale * show_sigma_scale * min(1.45, 0.85 + mag_delta * 0.10)
                brightness = star_brightness_scale * show_brightness_scale * 95.0 * (2.512 ** (mag_delta * 0.34))
                draw_gaussian_spot(star_map, x, y, brightness, sigma)

    if add_noise:
        star_map += np.random.normal(noise_mean, noise_std, (img_height, img_width))

    final_img_gray = np.clip(star_map, 0, 255).astype(np.uint8)
    if return_gray:
        return final_img_gray
    return cv2.cvtColor(final_img_gray, cv2.COLOR_GRAY2BGR)


def create_motion_star_image_by_fov(
        ra_in,
        dec_in,
        rotation_deg,
        fov_deg,
        img_width=1920,
        img_height=1080,
        limiting_mag=7.0,
        csv_path=None,
        add_noise=False,
        noise_mean=12,
        noise_std=6,
        return_gray=False,
        random_seed=None,
        motion_angle_deg=None,
        speed=1,
        show_star=2):
    """Create a Gaia star chart with one-direction dynamic motion blur.

    Args match create_star_image_by_fov. The blur direction is random by default.
    Bright stars produce longer trails than faint stars. speed can be 1, 2 or 3.
    show_star can be 1, 2, 3 or 4.
    """
    ra_in = float(ra_in)
    dec_in = float(dec_in)
    rotation_deg = float(rotation_deg)
    fov_deg = float(fov_deg)

    if not 0.0 <= ra_in <= 360.0:
        raise ValueError('ra_in must be in [0, 360] degrees')
    if not -90.0 <= dec_in <= 90.0:
        raise ValueError('dec_in must be in [-90, 90] degrees')
    if not 0.0 <= rotation_deg <= 360.0:
        raise ValueError('rotation_deg must be in [0, 360] degrees')
    if not 3.0 <= fov_deg <= 40.0:
        raise ValueError('fov_deg must be in [3, 40] degrees')
    if img_width <= 0 or img_height <= 0:
        raise ValueError('img_width and img_height must be positive')
    if speed not in (1, 2, 3):
        raise ValueError('speed must be 1, 2 or 3')

    ra_in, dec_in = _normalize_center_radec(ra_in, dec_in)
    rotation_deg = rotation_deg % 360.0
    draw_limit_mag = min(float(limiting_mag), 7.0)
    speed_scale = {1: 1.0, 2: 1.7, 3: 2.4}[speed]
    show_brightness_scale, show_sigma_scale = _star_display_scales(show_star)

    if csv_path is None:
        csv_path = _default_gaia_catalogue_path()
    csv_path = os.path.abspath(csv_path)
    mags_all, stars_3d_all, tree = _load_gaia_catalogue(csv_path)

    rng = np.random.default_rng(random_seed)
    if motion_angle_deg is None:
        motion_angle_rad = rng.uniform(0.0, 2.0 * np.pi)
    else:
        motion_angle_rad = np.radians(float(motion_angle_deg))
    motion_direction = np.array([np.cos(motion_angle_rad), np.sin(motion_angle_rad)])
    image_scale = np.hypot(img_width, img_height) / np.hypot(640.0, 360.0)

    fov_rad = np.radians(fov_deg)
    f_px = (img_width / 2.0) / np.tan(fov_rad / 2.0)
    K = np.array([
        [f_px, 0.0, img_width / 2.0],
        [0.0, f_px, img_height / 2.0],
        [0.0, 0.0, 1.0]
    ])

    diag_half_px = np.hypot(img_width / 2.0, img_height / 2.0)
    diag_fov_rad = 2.0 * np.arctan(diag_half_px / f_px)
    search_radius = 2.0 * np.sin(min(np.pi, diag_fov_rad * 1.08) / 2.0)

    z_c = _radec_to_unit_vector(ra_in, dec_in)
    indices = tree.query_ball_point(z_c, search_radius)
    star_map = np.zeros((img_height, img_width), dtype=np.float32)

    if indices:
        R_OBS_J2K = _create_observer_rotation(ra_in, dec_in, rotation_deg)
        local_stars_j2k = stars_3d_all[indices]
        local_mags = mags_all[indices]

        stars_obs = (R_OBS_J2K @ local_stars_j2k.T).T
        front_mask = stars_obs[:, 2] > 0
        stars_obs = stars_obs[front_mask]
        local_mags = local_mags[front_mask]

        if len(stars_obs) > 0:
            coords_homo = (K @ (stars_obs / stars_obs[:, 2:3]).T).T
            u, v = coords_homo[:, 0], coords_homo[:, 1]
            in_frame = (
                (u >= 0) & (u < img_width) &
                (v >= 0) & (v < img_height) &
                (local_mags <= draw_limit_mag)
            )
            u, v, local_mags = u[in_frame], v[in_frame], local_mags[in_frame]

            for x, y, mag in zip(u, v, local_mags):
                mag_delta = max(0.0, draw_limit_mag - mag)
                sigma = show_sigma_scale * min(1.45, 0.85 + mag_delta * 0.10)
                brightness = show_brightness_scale * 95.0 * (2.512 ** (mag_delta * 0.34))
                trail_length = speed_scale * image_scale * (2.0 + min(7.0, mag_delta) * 2.6)
                draw_motion_blurred_gaussian_spot(
                    star_map,
                    x,
                    y,
                    brightness,
                    sigma,
                    motion_direction,
                    trail_length
                )

    if add_noise:
        star_map += np.random.normal(noise_mean, noise_std, (img_height, img_width))

    final_img_gray = np.clip(star_map, 0, 255).astype(np.uint8)
    if return_gray:
        return final_img_gray
    return cv2.cvtColor(final_img_gray, cv2.COLOR_GRAY2BGR)


def create_polluted_star_image_by_fov(
        ra_in,
        dec_in,
        rotation_deg,
        fov_deg,
        img_width=1920,
        img_height=1080,
        limiting_mag=7.0,
        csv_path=None,
        add_noise=True,
        noise_mean=1.5,
        noise_std=5.5,
        return_gray=False,
        random_seed=None,
        patch_count=None,
        show_star=1):
    """Create a Gaia star chart with broad, blurred light-pollution patches."""
    gray_img = create_star_image_by_fov(
        ra_in,
        dec_in,
        rotation_deg,
        fov_deg,
        img_width=img_width,
        img_height=img_height,
        limiting_mag=limiting_mag,
        csv_path=csv_path,
        add_noise=False,
        return_gray=True,
        show_star=show_star
    )

    polluted_gray = apply_light_pollution(
        gray_img,
        random_seed=random_seed,
        patch_count=patch_count,
        add_noise=add_noise,
        noise_mean=noise_mean,
        noise_std=noise_std
    )

    if return_gray:
        return polluted_gray
    return cv2.cvtColor(polluted_gray, cv2.COLOR_GRAY2BGR)
