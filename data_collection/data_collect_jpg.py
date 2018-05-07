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
import winsound

import cv2

from data_collection.gamepad_cap import Gamepad
from data_collection.img_process import img_process

lock = threading.Lock()

# files to save training data
path = "E:\Graduation_Project"
img = "img\img{}.jpg"
img_path = os.path.join(path, "img\img{}.jpg")
table = 'training_data.csv'
# read previously stored data to avoid overwriting
img_num = 1
if os.path.isfile(os.path.join(path, table)):
    with open(os.path.join(path, table), 'r') as f:
        img_num = int(re.findall('\d+', f.readlines()[-1])[0]) + 1


def save(data):
    global img_num

    with lock:  # make sure that data is consistent
        # last_time = time.time()
        with open(os.path.join(path, table), 'a', newline='') as f:
            writer = csv.writer(f)

            for td in data:
                # write in csv: image_name, throttle, steering
                writer.writerow([img.format(img_num), td[1], td[2], td[3], td[4]])
                # write captures in files
                cv2.imwrite(img_path.format(img_num), td[0])
                img_num += 1
        # print('Saving took {} seconds'.format(time.time() - last_time))


def main():
    # initialize gamepad
    gamepad = Gamepad()
    gamepad.open()

    # last_time = time.time()  # to measure the number of frames
    alert_time = time.time()  # to signal about exceeding speed limit
    close = False  # to exit execution
    pause = True  # to pause execution
    training_data = []  # list for storing training data

    print("Press RB on your gamepad to start recording")
    while not close:
        while not pause:
            # read throttle and steering values from the gamepad
            throttle, steering = gamepad.get_state()
            # get screen, speed and direction
            screen, speed, direction = img_process("Grand Theft Auto V")

            training_data.append([screen, throttle, steering, speed, direction])

            if speed > 60 and time.time() - alert_time > 1:
                winsound.PlaySound('.\\resources\\alert.wav', winsound.SND_ASYNC)
                alert_time = time.time()

            # save the data every 500 iterations
            if len(training_data) % 500 == 0:
                threading.Thread(target=save, args=(training_data,)).start()
                training_data = []

            time.sleep(0.02)  # in order to slow down fps
            # print('Main loop took {} seconds'.format(time.time() - last_time))
            # last_time = time.time()

            if gamepad.get_RB():
                pause = True
                print('Paused. To exit the program press LB.')
                time.sleep(0.5)

        if gamepad.get_RB():
            pause = False
            print('Unpaused')
            time.sleep(1)
        elif gamepad.get_LB():
            gamepad.close()
            close = True
            print('Saving data and closing the program.')
            save(training_data)


if __name__ == '__main__':
    main()
