"""
Module for reading information from an Xbox gamepad
"""

import threading

from inputs import get_gamepad

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
        self.y_axisP = 0
        self.y_axisN = 0
        self.RB = 0
        self.LB = 0
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
                    self.y_axisP = event.state
                elif event.code == "ABS_Z":
                    self.y_axisN = -event.state
                elif event.code == "BTN_TR":
                    self.RB = event.state
                elif event.code == "BTN_TL":
                    self.LB = event.state
                else:
                    pass  # we're not interested in the remain signals

    def get_state(self):
        xAxis = self.x_axis
        yAxis = self.y_axisP if self.y_axisP > 60 else self.y_axisN

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

    def get_RB(self):
        return self.RB

    def get_LB(self):
        return self.LB

    def close(self):
        self.stop = True
