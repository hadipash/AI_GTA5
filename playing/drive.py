"""
Car driving module.
"""

# reading and writing files
import os
import time

import cv2
# matrix math
import numpy as np
# load our saved model
from keras.models import load_model

# helper classes
from data_collection.keycap import key_check
from data_collection.screencap import grab_screen
from playing.keycontrol import *
from training.utils import preprocess

# init our model
model = None
path = "..\\training"


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
            # apply the preprocessing
            image = preprocess(cv2.resize(grab_screen("Grand Theft Auto V"), (320, 240)))
            image = np.array([image])  # the model expects 4D array

            # predict the steering angle for the image
            controls = model.predict(image, batch_size=1)
            controls = [round(float(controls[0][0])), round(float(controls[0][1]))]
            send_control(controls)

            keys = key_check()
            if 'T' in keys:
                pause = True
                print('Paused. To exit the program press Z.')
                time.sleep(0.5)

        keys = key_check()
        if 'T' in keys:
            pause = False
            print('Unpaused')
            time.sleep(1)
        elif 'Z' in keys:
            close = True
            print('Closing the program.')


def main():
    # load model
    global model
    model_num = int(input('Input a model number: '))
    location = os.path.join(path, 'model-{0:03d}.h5'.format(model_num))
    model = load_model(location)
    # control a car
    drive()


if __name__ == '__main__':
    main()
