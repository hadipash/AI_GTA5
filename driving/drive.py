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

from data_collection.img_process import grab_screen
# helper classes
from data_collection.key_cap import key_check
# gamepad axes limits and gamepad module
from driving.gamepad import AXIS_MIN, AXIS_MAX, TRIGGER_MAX, XInputDevice
from training.utils import preprocess

gamepad = None

# init the model
model = None
path = "..\\training"

# set scales for axes
SCALE_AXIS_MIN = AXIS_MIN / -10
SCALE_AXIS_MAX = AXIS_MAX / 10
SCALE_TRIGGER = TRIGGER_MAX / 10


def set_gamepad(controls):
    # trigger value
    trigger = int(round(controls[0][0] * SCALE_TRIGGER))
    if trigger >= 0:
        # left trigger is zero
        gamepad.SetTrigger('L', 0)
        gamepad.SetTrigger('R', trigger)
    else:
        # inverse value
        trigger = -trigger
        # right trigger is zero
        gamepad.SetTrigger('L', trigger)
        gamepad.SetTrigger('R', 0)

    # axis value
    axis = 0
    if controls[0][1] >= 0:
        axis = int(round(controls[0][1] * SCALE_AXIS_MAX))
    else:
        axis = int(round(controls[0][1] * SCALE_AXIS_MIN))
    gamepad.SetAxis('X', axis)


def drive():
    global gamepad
    gamepad = XInputDevice(1)
    gamepad.PlugIn()

    # countdown for having time to open GTA V window
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)
    print("Start!")

    # last_time = time.time()  # to measure the number of frames
    close = False  # to exit execution
    pause = False  # to pause execution

    while not close:
        while not pause:
            # apply the preprocessing
            image = preprocess(cv2.resize(grab_screen("Grand Theft Auto V"), (320, 240)))
            image = np.array([image])  # the model expects 4D array

            # predict the throttle and steering angle for the image
            controls = model.predict(image, batch_size=1)
            # set the gamepad values
            set_gamepad(controls)
            # print("Throttle: {0:02d}, Steering: {0:02d}".format(int(round(controls[0][0])),
            #       int(round(controls[0][1]))))

            # print('Main loop took {} seconds'.format(time.time() - last_time))
            # last_time = time.time()

            keys = key_check()
            if 'T' in keys:
                pause = True
                # release gamepad keys
                gamepad.SetTrigger('L', 0)
                gamepad.SetTrigger('R', 0)
                gamepad.SetAxis('X', 0)
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
            gamepad.UnPlug()


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
