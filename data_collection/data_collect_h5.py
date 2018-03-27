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

from data_collection.keycap import key_check
from data_collection.screencap import grab_screen

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

lock = threading.Lock()

# open the data file
data_file = None
if os.path.isfile("training_data.h5"):
    data_file = h5py.File("training_data.h5", 'a')
else:
    data_file = h5py.File("training_data.h5", 'w')
    # Write data in chunks for faster writing and reading by NN
    data_file.create_dataset('img', (0, 240, 320, 3), dtype='u1',
                             maxshape=(None, 240, 320, 3), chunks=(300, 240, 320, 3))
    data_file.create_dataset('key', (0, 9), dtype='u1',
                             maxshape=(None, 9), chunks=(300, 9))


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
    elif 'W' in keys:
        output = w
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    elif 'S' in keys:
        output = s
    else:
        output = nk

    return output


def save(data_img, data_key):
    if data_img:  # if the list is not empty
        with lock:  # make sure that data is consistent
            # last_time = time.time()
            data_file["img"].resize((data_file["img"].shape[0] + len(data_img)), axis=0)
            data_file["img"][-len(data_img):] = data_img
            data_file["key"].resize((data_file["key"].shape[0] + len(data_key)), axis=0)
            data_file["key"][-len(data_key):] = data_key
            # print('Saving took {} seconds'.format(time.time() - last_time))


def main():
    # countdown for having time to open GTA V window
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)
    print("Start!")

    # last_time = time.time()     # to measure the number of frames
    close = False  # to exit execution
    pause = False  # to pause execution
    training_img = []  # lists for storing training data
    training_key = []

    while not close:
        while not pause:
            screen = cv2.resize(grab_screen("Grand Theft Auto V"), (320, 240))
            output = keys_to_output(key_check())
            training_img.append(screen)
            training_key.append(output)

            # save the data every 300 iterations
            if len(training_img) % 300 == 0:
                threading.Thread(target=save, args=(training_img, training_key)).start()
                training_img = []
                training_key = []

            time.sleep(0.02)  # in order to slow down fps
            # print('Main loop took {} seconds'.format(time.time() - last_time))
            # last_time = time.time()

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
            save(training_img, training_key)

    data_file.close()


if __name__ == '__main__':
    main()
