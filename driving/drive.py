"""
Car driving module.
"""

# reading and writing files
import os
import time

import cv2
import numpy as np
# load our saved model
from keras.models import load_model

# helper classes
from data_collection.img_process import img_process
from data_collection.key_cap import key_check
# gamepad axes limits and gamepad module
from driving.gamepad import AXIS_MIN, AXIS_MAX, TRIGGER_MAX, XInputDevice
from object_detection.direction import Direct
# YOLO algorithm
from object_detection.object_detect import yolo_detection
# lane detection algorithm
from object_detection.lane_detect import detect_lane, draw_lane
from training.utils import preprocess

model_path = "..\\training"
gamepad = None


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


def drive(model):
    global gamepad
    gamepad = XInputDevice(1)
    gamepad.PlugIn()

    # last_time = time.time()  # to measure the number of frames
    close = False  # to exit execution
    pause = True  # to pause execution
    stop = False    # to stop the car
    throttle = 0
    left_line_max = 75
    right_line_max = 670

    print("Press T to start driving")

    while not close:
        yolo_screen, resized, speed, direct = img_process("Grand Theft Auto V")
        cv2.imshow("Driving-mode", yolo_screen)
        cv2.waitKey(1)

        while not pause:
            # apply the preprocessing
            screen, resized, speed, direct = img_process("Grand Theft Auto V")
            radar = cv2.cvtColor(resized[206:226, 25:45, :], cv2.COLOR_RGB2BGR)[:, :, 2:3]
            resized = preprocess(resized)
            left_line_color = [0, 255, 0]
            right_line_color = [0, 255, 0]

            # predict steering angle for the image
            # original + radar (small) + speed
            controls = model.predict([np.array([resized]), np.array([radar]), np.array([speed])], batch_size=1)
            # check that the car is following lane
            lane, stop_line = detect_lane(screen)
            # detect objects
            yolo_screen, color_detected, obj_distance = yolo_detection(screen, direct)

            if not stop:
                # adjusting speed
                if speed < 45:
                    throttle = 0.4
                elif speed > 50:
                    throttle = 0.0

                if 0 <= obj_distance <= 0.6:
                    if speed < 5:
                        throttle = 0
                    else:
                        throttle = -0.7 if obj_distance <= 0.4 else -0.3

                elif color_detected == "Red":
                    if stop_line:
                        if speed < 5:
                            throttle = 0
                        elif 0 <= stop_line[0][1] <= 50:
                            throttle = -0.5
                        elif 50 < stop_line[0][1] <= 120:
                            throttle = -1
                    # else:
                    #     throttle = -0.5
            elif speed > 5:
                throttle = -1
            else:
                throttle = 0
                cv2.destroyAllWindows()
                pause = True

            # adjusting steering angle
            if lane[0] and lane[0][0] > left_line_max:
                if abs(controls[0][0]) < 0.27:
                    controls[0][0] = 0.27
                    left_line_color = [0, 0, 255]
            elif lane[1] and lane[1][0] < right_line_max:
                if abs(controls[0][0]) < 0.27:
                    controls[0][0] = -0.27
                    right_line_color = [0, 0, 255]

            # set the gamepad values
            set_gamepad([[controls[0][0], throttle]])

            # print('Main loop took {} seconds'.format(time.time() - last_time))
            # last_time = time.time()

            screen[280:-130, :, :] = draw_lane(screen[280:-130, :, :], lane, stop_line,
                                               left_line_color, right_line_color)
            cv2.imshow("Driving-mode", yolo_screen)
            cv2.waitKey(1)

            if direct == 6:
                print("Arrived at destination.")
                stop = True

            # print('Main loop took {} seconds'.format(time.time() - last_time))
            # last_time = time.time()

            keys = key_check()
            if 'T' in keys:
                cv2.destroyAllWindows()
                pause = True
                # release gamepad keys
                set_gamepad([[0, 0]])
                print('Paused. To exit the program press Z.')
                time.sleep(0.5)

        keys = key_check()
        if 'T' in keys:
            pause = False
            stop = False
            print('Unpaused')
            time.sleep(1)
        elif 'Z' in keys:
            cv2.destroyAllWindows()
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
