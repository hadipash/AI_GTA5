"""
Data collection module (saves data in CSV and JPG formats).
Saves screen captures and pressed keys into a file
for further trainings of NN.
"""

import csv
import os
import re
import threading
import time

import cv2

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

# files to save training data
img = "img/img{}.jpg"
table = 'training_data.csv'
# read previously stored data to avoid overwriting
img_num = 1
if os.path.isfile(table):
    with open(table, 'r') as f:
        img_num = int(re.findall('\d+', f.readlines()[-1])[0]) + 1


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


def save(data):
    global img_num

    with lock:  # make sure that data is consistent
        # last_time = time.time()
        with open(table, 'a', newline='') as f:
            writer = csv.writer(f)

            for td in data:
                # write in csv
                writer.writerow([img.format(img_num), td[1]])
                # write captures in files
                cv2.imwrite(img.format(img_num), td[0])
                img_num += 1
        # print('Saving took {} seconds'.format(time.time() - last_time))


def main():
    # countdown for having time to open GTA V window
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)
    print("Start!")

    # last_time = time.time()  # to measure the number of frames
    close = False  # to exit execution
    pause = False  # to pause execution
    training_data = []  # list for storing training data

    while not close:
        while not pause:
            screen = cv2.resize(grab_screen("Grand Theft Auto V"), (320, 240))
            output = keys_to_output(key_check())
            training_data.append([screen, output])

            # save the data every 500 iterations
            if len(training_data) % 500 == 0:
                threading.Thread(target=save, args=(training_data,)).start()
                training_data = []

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
            save(training_data)


if __name__ == '__main__':
    main()
