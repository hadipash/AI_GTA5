# This code based on Harrison Kinsley's (Sentdex) code (https://github.com/Sentdex/pygta5)

"""
Data collection module (saves data in H5 format).
Saves screen captures and pressed keys into a file
for further trainings of NN.
"""

import os
import threading
import time

import cv2
import h5py

from data_collection.keycap import key_check, Gamepad
from data_collection.screencap import grab_screen

lock = threading.Lock()

# open the data file
path = "E:\Graduation_Project"
data_file = None
if os.path.isfile(os.path.join(path, "training_data.h5")):
    data_file = h5py.File(os.path.join(path, "training_data.h5"), 'a')
else:
    data_file = h5py.File(os.path.join(path, "training_data.h5"), 'w')
    # Write data in chunks for faster writing and reading by NN
    data_file.create_dataset('img', (0, 240, 320, 3), dtype='u1',
                             maxshape=(None, 240, 320, 3), chunks=(30, 240, 320, 3))
    data_file.create_dataset('controls', (0, 2), dtype='i1', maxshape=(None, 2), chunks=(30, 2))


# in case of using a keyboard
def keys_to_output(keys):
    # initial values: no key pressed
    throttle = 0
    steering = 0

    if 'W' in keys:
        throttle = 1
    elif 'S' in keys:
        throttle = -1

    if 'A' in keys:
        steering = -1
    elif 'D' in keys:
        steering = 1

    return throttle, steering


def save(data_img, controls):
    if data_img:  # if the list is not empty
        with lock:  # make sure that data is consistent
            # last_time = time.time()
            data_file["img"].resize((data_file["img"].shape[0] + len(data_img)), axis=0)
            data_file["img"][-len(data_img):] = data_img
            data_file["controls"].resize((data_file["controls"].shape[0] + len(controls)), axis=0)
            data_file["controls"][-len(controls):] = controls
            # print('Saving took {} seconds'.format(time.time() - last_time))


def main():
    # initialize gamepad
    gamepad = Gamepad()
    gamepad.open()

    # countdown for having time to open GTA V window
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)
    print("Start!")

    # last_time = time.time()     # to measure the number of frames
    close = False  # to exit execution
    pause = False  # to pause execution
    training_img = []  # lists for storing training data
    controls = []

    while not close:
        while not pause:
            screen = cv2.resize(grab_screen("Grand Theft Auto V"), (320, 240))
            # read throttle and steering values from the keyboard
            # throttle, steering = keys_to_output(key_check())
            # read throttle and steering values from the gamepad
            throttle, steering = gamepad.get_state()
            training_img.append(screen)
            controls.append([throttle, steering])

            # save the data every 30 iterations
            if len(training_img) % 30 == 0:
                threading.Thread(target=save, args=(training_img, controls)).start()
                training_img = []
                controls = []

            time.sleep(0.02)  # in order to slow down fps
            # print('Main loop took {} seconds'.format(time.time() - last_time))
            # last_time = time.time()

            keys = key_check()
            if 'T' in keys:
                gamepad.close()
                pause = True
                print('Paused. To exit the program press Z.')
                time.sleep(0.5)

        keys = key_check()
        if 'T' in keys:
            gamepad.open()
            pause = False
            print('Unpaused')
            time.sleep(1)
        elif 'Z' in keys:
            close = True
            print('Saving data and closing the program.')
            save(training_img, controls)

    data_file.close()


if __name__ == '__main__':
    main()
