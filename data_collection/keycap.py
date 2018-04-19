# This code based on Harrison Kinsley's (Sentdex) code (https://github.com/Sentdex/pygta5)
# Citation: Box Of Hats (https://github.com/Box-Of-Hats)

"""
Module for reading keys from keyboard or information from an Xbox gamepad
"""

import threading
import win32api as wapi

from inputs import get_gamepad

# Keyboard part
keyList = ["\b"]
 
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)


def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


# Gamepad part
AXIS_MAX = 32767
AXIS_MIN = -32768
TRIGGER_MAX = 255
TRIGGER_MIN = -255

AXIS_MAX_NORM = 10 / AXIS_MAX
AXIS_MIN_NORM = -10 / AXIS_MIN
TRIGGER_MAX_NORM = 10 / TRIGGER_MAX
TRIGGER_MIN_NORM = -10 / TRIGGER_MIN

DEADZONE = 3


class Gamepad:
    def __init__(self):
        self.x_axis = 0
        self.y_axis = 0
        self.stop = False

    def open(self):
        self.stop = False
        threading.Thread(target=self.run).start()

    def run(self):
        while not self.stop:
            events = get_gamepad()
            for event in events:
                if event.code == "ABS_X":
                    self.x_axis = event.state
                elif event.code == "ABS_RZ":
                    self.y_axis = event.state
                elif event.code == "ABS_Z":
                    self.y_axis = -event.state
                else:
                    pass  # we're not interested in the remain signals

    def get_state(self):
        xAxis = self.x_axis
        yAxis = self.y_axis

        # normalize x axis
        if xAxis > 0:
            xAxis = int(round(xAxis * AXIS_MAX_NORM))
        else:
            xAxis = int(round(xAxis * AXIS_MIN_NORM))
        if -DEADZONE < xAxis < DEADZONE:
            xAxis = 0
        # normalize y axis
        if yAxis > 0:
            yAxis = int(round(yAxis * TRIGGER_MAX_NORM))
        else:
            yAxis = int(round(yAxis * TRIGGER_MIN_NORM))
        if -DEADZONE < yAxis < DEADZONE:
            yAxis = 0

        # return throttle and then steering
        return yAxis, xAxis

    def close(self):
        self.stop = True
