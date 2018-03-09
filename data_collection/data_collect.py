"""
Data collection module.
Saves screen captures and pressed keys into a file
for further trainings of NN.
"""

import numpy as np
import time
import os
import cv2

from data_collection.screencap import grab_screen
from data_collection.keycap import key_check

# key values in One-Hot Encoding
w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]  # no key pressed

# file to save training data in
file_name = 'training_data.npy'
if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []


def keys_to_output(keys):
    """
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    """
    output = []

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk

    return output


def main():
    # countdown for having time to open GTA V window
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)

    last_time = time.time()  # to measure the number of frames
    close = False  # to exit execution
    pause = False  # to pause execution
    img_num = 1  # in case of writing captures in files

    while not close:
        while not pause:
            screen = cv2.resize(grab_screen("Grand Theft Auto V"), (320, 228))
            output = keys_to_output(key_check())
            training_data.append([screen, output])

            # write capture in a file
            cv2.imwrite("img{}.jpg".format(img_num), screen)
            img_num += 1

            # save the data every 1000 iterations
            if len(training_data) % 1000 == 0:
                print(len(training_data))
                np.save(file_name, training_data)

            time.sleep(0.02)  # in order to slow down fps
            print('loop took {} seconds'.format(time.time() - last_time))
            last_time = time.time()

            keys = key_check()
            if 'T' in keys:
                pause = True
                print('Paused. To exit the program press Q.')
                time.sleep(1)

        keys = key_check()
        if 'T' in keys:
            pause = False
            print('Unpaused')
            time.sleep(1)
        elif 'Q' in keys:
            close = True
            print('Saving data and closing the program.')
            np.save(file_name, training_data)


if __name__ == '__main__':
    main()
