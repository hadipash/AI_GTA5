# This code based on Musi13's code (https://github.com/Musi13/pyvxbox)

"""
Gamepad emulating module.
"""

import sys
from ctypes import *

dll_path = "vXboxInterface.dll"

try:
    _vx = cdll.LoadLibrary(dll_path)
except OSError as e:
    print(e)
    sys.exit("Unable to load vXbox SDK DLL. Ensure that %s is present" % dll_path)

if not _vx.isVBusExists():
    raise Exception('Xbox VBus does not exist')

AXIS_MAX = 32767
AXIS_MIN = -32768
TRIGGER_MAX = 255
BTN_ON = True
BTN_OFF = False


class XInputDevice:
    def __init__(self, port):
        if _vx.isControllerExists(port):
            raise Exception('Port %d is already used' % port)
        self.UserIndex = port

    def PlugIn(self):
        _vx.PlugIn(self.UserIndex)

    def UnPlug(self, force=False):
        if not force:
            _vx.UnPlug(self.UserIndex)
        else:
            _vx.UnPlugForce(self.UserIndex)

    def SetBtn(self, button, value):
        function = {
            'A': _vx.SetBtnA,
            'B': _vx.SetBtnB,
            'X': _vx.SetBtnX,
            'Y': _vx.SetBtnY,
            'Start': _vx.SetBtnStart,
            'Back': _vx.SetBtnBack,
            'LT': _vx.SetBtnLT,
            'RT': _vx.SetBtnRT,
            'LB': _vx.SetBtnLB,
            'RB': _vx.SetBtnRB,
            'GD': _vx.SetBtnGD
        }.get(button, None)
        if function is None:
            raise Exception('Unknown button %s' % str(button))
        function(self.UserIndex, value)

    def SetTrigger(self, trigger, value):
        function = {
            'L': _vx.SetTriggerL,
            'R': _vx.SetTriggerR
        }.get(trigger, None)
        if function is None:
            raise Exception('Unknown trigger %s' % str(trigger))
        function(self.UserIndex, value)

    def SetAxis(self, axis, value):
        function = {
            'X': _vx.SetAxisX,
            'Y': _vx.SetAxisY,
            'Rx': _vx.SetAxisRx,
            'Ry': _vx.SetAxisRy
        }.get(axis, None)
        if function is None:
            raise Exception('Unknown axis %s' % str(axis))
        function(self.UserIndex, value)

    def SetDpad(self, direction, value=0):
        function = {
            'Up': _vx.SetDpadUp,
            'Right': _vx.SetDpadRight,
            'Down': _vx.SetDpadDown,
            'Left': _vx.SetDpadLeft,
            '': _vx.SetDpad
        }.get(direction, None)
        if function is None:
            raise Exception('Unknown direction %s' % str(direction))
        if direction == '':
            function(self.UserIndex, value)
        else:
            function(self.UserIndex)

    def GetLedNumber(self, pLed):
        _vx.GetLedNumber(self.UserIndex, pLed)

    def GetVibration(self, pVib):
        _vx.GetVibration(self.UserIndex, pVib)
