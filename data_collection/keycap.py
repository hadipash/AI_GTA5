# This code based on Harrison Kinsley's (Sentdex) code (https://github.com/Sentdex/pygta5)
# Citation: Box Of Hats (https://github.com/Box-Of-Hats)

"""
Module for reading keys
"""

import threading
import win32api as wapi

from inputs import get_gamepad

keyList = ["\b"]
 
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)


def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


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
        return xAxis, yAxis

    def close(self):
        self.stop = True
