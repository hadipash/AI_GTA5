"""
Car driving module.
"""

# parsing command line arguments
import argparse
# reading and writing files
import os
# high level file operations
import shutil
import time
# for frametimestamp saving
from datetime import datetime

import cv2
# matrix math
import numpy as np
# load our saved model
from keras.models import load_model

from data_collection.keycap import key_check
from data_collection.screencap import grab_screen
from playing.keycontrol import *
# helper classes
from training.utils import preprocess

# init our model and image array as empty
model = None
args = None


def send_control(controls):
    if controls == [1, 0]:
        straight()
    elif controls == [1, -1]:
        forward_left()
    elif controls == [1, 1]:
        forward_right()
    elif controls == [0, -1]:
        left()
    elif controls == [0, 1]:
        right()
    elif controls == [0, 0]:
        no_keys()
    elif controls == [-1, 0]:
        reverse()
    elif controls == [-1, -1]:
        reverse_left()
    else:
        reverse_right()


def drive():
    # countdown for having time to open GTA V window
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)
    print("Start!")

    close = False  # to exit execution
    pause = False  # to pause execution

    while not close:
        while not pause:
            # TODO: add speed and direction into input data

            # apply the preprocessing
            image = preprocess(cv2.resize(grab_screen("Grand Theft Auto V"), (320, 240)))
            image = np.array([image])  # the model expects 4D array

            # predict the steering angle for the image
            controls = round(float(model.predict(image, batch_size=1)))
            send_control(controls)

            # save frame
            if args.image_folder != '':
                timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                image_filename = os.path.join(args.image_folder, timestamp)
                image.save('{}.jpg'.format(image_filename))

            keys = key_check()
            if 'T' in keys:
                pause = True
                print('Paused. To exit the program press Q.')
                time.sleep(0.5)

        keys = key_check()
        if 'T' in keys:
            pause = False
            print('Unpaused')
            time.sleep(1)
        elif 'Q' in keys:
            close = True
            print('Saving data and closing the program.')


def main():
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        default=os.path.join("D:\Projects\Python\AI GTA5\\training", 'model-010.h5'),
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    global args
    args = parser.parse_args()

    # load model
    global model
    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    drive()


if __name__ == '__main__':
    main()
