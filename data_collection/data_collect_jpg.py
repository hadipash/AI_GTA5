"""
Data collection module.
Saves screen captures and pressed keys into a file
for further trainings of NN.
"""

import csv
import time
import cv2
import os

from numpy import genfromtxt
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

# files to save training data
img = "img/img{}.jpg"
table = 'training_data.csv'

# read previously stored data to avoid overwriting
if os.path.isfile(table):
    with open(table, 'rb') as f:
        lines = f.readlines()
        genfromtxt(lines[-1:], delimiter=',')
else:
    img_num = 1

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


def save():
    global img_num, training_data

    with open(table, 'a', newline='') as f:
        writer = csv.writer(f)

        for td in training_data:
            writer.writerow([img.format(img_num), td[1]])
            # write captures in files
            cv2.imwrite(img.format(img_num), td[0])
            img_num += 1

    training_data = []  # clear temporary array of data


def main():
    # countdown for having time to open GTA V window
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)
    print("Start!")

    last_time = time.time()  # to measure the number of frames
    close = False  # to exit execution
    pause = False  # to pause execution

    while not close:
        while not pause:
            screen = cv2.resize(grab_screen("Grand Theft Auto V"), (320, 240))
            output = keys_to_output(key_check())
            training_data.append([screen, output])

            # save the data every 100 iterations
            if len(training_data) % 100 == 0:
                print("Saving")
                save()
                print('loop took {} seconds'.format(time.time() - last_time))
            else:
                time.sleep(0.02)  # in order to slow down fps

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
            save()


if __name__ == '__main__':
    main()
