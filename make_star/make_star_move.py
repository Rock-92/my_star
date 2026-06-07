from make_star.image_generator import create_motion_star_image_by_fov


def create_star_image_move(
        ra,
        dec,
        rotation,
        fov,
        img_width=1920,
        img_height=1080,
        add_noise=False,
        speed=1,
        show_star=2,
        **kwargs):
    """Generate a motion-blurred Gaia star image.

    Args:
        ra (float): Center right ascension in degrees, in [0, 360].
        dec (float): Center declination in degrees, in [-90, 90].
        rotation (float): Roll angle in degrees, in [0, 360].
        fov (float): Horizontal field of view in degrees, in [3, 40].
        img_width (int): Output image width in pixels.
        img_height (int): Output image height in pixels.
        add_noise (bool): Whether to add Gaussian background noise.
        speed (int): Motion blur speed level. Must be 1, 2 or 3.
        show_star (int): Star display level. Must be 1, 2, 3 or 4.
        **kwargs: Optional arguments passed to create_motion_star_image_by_fov,
            such as limiting_mag, noise_mean, noise_std, return_gray,
            random_seed and motion_angle_deg.

    Returns:
        numpy.ndarray: Generated motion-blurred star image.
    """
    return create_motion_star_image_by_fov(
        ra,
        dec,
        rotation,
        fov,
        img_width=img_width,
        img_height=img_height,
        add_noise=add_noise,
        speed=speed,
        show_star=show_star,
        **kwargs
    )


create_star_image_move = create_star_image_move
