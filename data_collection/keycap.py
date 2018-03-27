# This code based on Harrison Kinsley's (Sentdex) code (https://github.com/Sentdex/pygta5)
# Citation: Box Of Hats (https://github.com/Box-Of-Hats)

import win32api as wapi
 
keyList = ["\b"]
 
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)


def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys
