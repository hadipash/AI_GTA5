# This code based on Siraj Raval's code (https://github.com/llSourcell/How_to_simulate_a_self_driving_car)

import math

import cv2
import numpy as np
import tensorflow as tf

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
RADAR_HEIGHT, RADAR_WIDTH, RADAR_CHANNELS = 20, 20, 1
RADAR_SHAPE = (RADAR_HEIGHT, RADAR_WIDTH, RADAR_CHANNELS)


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[90:-50, :, :]


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


# def choose_image(data_dir, center, left, right, steering_angle):
#     """
#     Randomly choose an image from the center, left or right, and adjust
#     the steering angle.
#     """
#     choice = np.random.choice(3)
#     if choice == 0:
#         return load_image(data_dir, left), steering_angle + 0.2
#     elif choice == 1:
#         return load_image(data_dir, right), steering_angle - 0.2
#     return load_image(data_dir, center), steering_angle


# flip image causes car riding on the opposite direction lane
# def random_flip(image, steering_angle):
#     """
#     Randomly flip the image left <-> right, and adjust the steering angle.
#     """
#     if np.random.rand() < 0.5:
#         image = cv2.flip(image, 1)
#         steering_angle = -steering_angle
#     return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image vertically and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)

    # adjusting steering angle
    t_x = trans_x / 25
    if t_x > 0:
        t_x = math.ceil(t_x)
        if t_x > 2:
            steering_angle += (t_x - 2)
            if steering_angle > 10:
                steering_angle = 10
    else:
        t_x = math.floor(t_x)
        if t_x < -2:
            steering_angle += (t_x + 2)
            if steering_angle < -10:
                steering_angle = -10

    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    # apply an affine transformation to an image
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(image, steering_angle, range_x=250, range_y=20):
    """
    Generate an augmented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    # image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    # image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


def batch_generator(data, indexes, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    # preprocessing on the CPU
    with tf.device('/cpu:0'):
        images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
        radars = np.empty([batch_size, RADAR_HEIGHT, RADAR_WIDTH, RADAR_CHANNELS])
        # metrics = np.empty([batch_size, 2])
        # controls = np.empty([batch_size, 2])
        speeds = np.empty(batch_size)
        controls = np.empty(batch_size)
        while True:
            i = 0
            for index in np.random.permutation(indexes):
                camera = data['img'][index]
                radar = cv2.cvtColor(camera[206:226, 25:45, :], cv2.COLOR_RGB2BGR)
                steer = data['controls'][index][1]

                # augmentation
                if is_training:
                    prob = np.random.rand()
                    if (abs(steer) < 0.4 and prob > 0.2) or (prob < 0.6):
                        camera, steer = augment(camera, steer)

                # add the image and steering angle to the batch
                images[i] = preprocess(camera)
                radars[i] = radar[:, :, 2:3]
                # controls[i] = [data['controls'][index][0] / 10, steer / 10]  # normalized throttle and steering
                controls[i] = steer / 10
                speeds[i] = data['metrics'][index][0]
                # metrics[i] = data['metrics'][index]
                i += 1
                if i == batch_size:
                    break
            # yield [images, metrics], controls
            yield [images, radars, speeds], controls
