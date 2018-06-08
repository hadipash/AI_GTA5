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
from data_collection.img_process import img_process
from data_collection.key_cap import key_check
# gamepad axes limits and gamepad module
from driving.gamepad import AXIS_MIN, AXIS_MAX, TRIGGER_MAX, XInputDevice
from training.utils import preprocess
# for using yolo
from darkflow.net.build import TFNet
from detect import yolo_detection

model_path = "..\\training"
gamepad = None

#set yolo option
option = {
    'model': '../cfg/yolo.cfg',
    'load': '../bin/yolov2.weights',
    'threshold': 0.3,
    'gpu':0.5
}
tfnet = TFNet(option)


def set_gamepad(controls):
    # trigger value
    trigger = int(round(controls[0][1] * TRIGGER_MAX))
    if trigger >= 0:
        # set left trigger to zero
        gamepad.SetTrigger('L', 0)
        gamepad.SetTrigger('R', trigger)
    else:
        # inverse value
        trigger = -trigger
        # set right trigger to zero
        gamepad.SetTrigger('L', trigger)
        gamepad.SetTrigger('R', 0)

    # axis value
    axis = 0
    if controls[0][0] >= 0:
        axis = int(round(controls[0][0] * AXIS_MAX))
    else:
        axis = int(round(controls[0][0] * (-AXIS_MIN)))
    gamepad.SetAxis('X', axis)


def stop():
    gamepad.SetTrigger('L', 0)
    gamepad.SetTrigger('R', 0)
    gamepad.SetAxis('X', 0)


def drive(model):
    global gamepad
    gamepad = XInputDevice(1)
    gamepad.PlugIn()

    # last_time = time.time()  # to measure the number of frames
    close = False  # to exit execution
    pause = True  # to pause execution
    throttle = 0

    print("Press T to start driving")
    while not close:
        while not pause:
            # apply the preprocessing
            image, speed, direct = img_process("Grand Theft Auto V")
            radar = cv2.cvtColor(image[206:226, 25:45, :], cv2.COLOR_RGB2BGR)[:, :, 2:3]
            image = preprocess(image)

            # predict steering angle for the image
            # original + radar (small) + speed
            controls = model.predict([np.array([image]), np.array([radar]), np.array([speed])], batch_size=1)

            if speed < 35:
                throttle = 0.4
                controls = [[controls[0][0], throttle]]
            elif speed > 40:
                throttle = 0.0
                controls = [[controls[0][0], throttle]]
            else:
                controls = [[controls[0][0], throttle]]

            # set the gamepad values
            set_gamepad(controls)
            # print("Steering: {0:.2f}".format(controls[0][0]))
			
			#for yolo detection
            screen = np.array(grab_screen("Grand Theft Auto V"), dtype=np.uint8)
            yolo_detection(tfnet, screen, speed, controls, gamepad, direct)
            cv2.imshow('frame', screen)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if direct == 6:
                print("Arrived at destination.")
                stop()
                pause = True

            # print('Main loop took {} seconds'.format(time.time() - last_time))
            # last_time = time.time()

            keys = key_check()
            if 'T' in keys:
                pause = True
                # release gamepad keys
                stop()
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
    location = os.path.join(model_path, 'base_model.h5')
    model = load_model(location)
    # control a car
    drive(model)


if __name__ == '__main__':
    main()
