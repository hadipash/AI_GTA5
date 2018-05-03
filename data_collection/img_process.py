"""
Module for preprocessing screen captures
"""

import win32gui
import win32ui

import cv2
import numpy as np
import win32con


def initKNN(fname):
    knn = cv2.ml.KNearest_create()
    with np.load(fname) as data:
        train = data['train']
        train_labels = data['train_labels']
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    return knn


knnDigits = initKNN('digit.npz')
knnArrows = initKNN('arrows.npz')


# Done by Frannecklp
def grab_screen(winName):
    desktop = win32gui.GetDesktopWindow()

    # get area by a window name
    gtawin = win32gui.FindWindow(None, winName)
    # get the bounding box of the window
    left, top, x2, y2 = win32gui.GetWindowRect(gtawin)
    # cut window boarders
    top += 32
    left += 3
    y2 -= 4
    x2 -= 4
    width = x2 - left + 1
    height = y2 - top + 1

    # the device context(DC) for the entire window (title bar, menus, scroll bars, etc.)
    hwindc = win32gui.GetWindowDC(desktop)
    # Create a DC object from an integer handle
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    # Create a memory device context that is compatible with the source DC
    memdc = srcdc.CreateCompatibleDC()
    # Create a bitmap object
    bmp = win32ui.CreateBitmap()
    # Create a bitmap compatible with the specified device context
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    # Select an object into the device context.
    memdc.SelectObject(bmp)
    # Copy a bitmap from the source device context to this device context
    # parameters: destPos, size, dc, srcPos, rop(the raster operation))
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    # the bitmap bits
    signedIntsArray = bmp.GetBitmapBits(True)
    # form a 1-D array initialized from text data in a string.
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    # Delete all resources associated with the device context
    srcdc.DeleteDC()
    memdc.DeleteDC()
    # Releases the device context
    win32gui.ReleaseDC(desktop, hwindc)
    # Delete the bitmap and freeing all system resources associated with the object.
    # After the object is deleted, the specified handle is no longer valid.
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)


def predict(img, knn):
    ret, result, neighbours, dist = knn.findNearest(img, k=1)
    return result


def preprocess(img, t_value, size):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thr = cv2.threshold(gray, t_value, 255, cv2.THRESH_BINARY)
    return thr.reshape(-1, size).astype(np.float32)


def convert_speed(num1, num2, num3):
    hundreds = 1
    tens = 1
    speed = 0

    if num3[0][0] != 11:
        hundreds = 10
        tens = 10
        speed += int(num3[0][0])
    if num2[0][0] != 11:
        speed += tens * int(num2[0][0])
        hundreds = tens * 10
    if num1[0][0] != 11:
        speed += hundreds * int(num1[0][0])

    return speed


def img_process(winName: str = "Grand Theft Auto V"):
    image = grab_screen(winName)

    # three fields for numbers
    num1 = preprocess(image[572:582, 680:689], 180, 90)
    num2 = preprocess(image[572:582, 688:697], 180, 90)
    num3 = preprocess(image[572:582, 695:704], 180, 90)
    # one field for direction arrows
    direct = preprocess(image[566:577, 17:28], 170, 121)

    # predict numbers and directions
    num1 = predict(num1, knnDigits)
    num2 = predict(num2, knnDigits)
    num3 = predict(num3, knnDigits)
    direct = predict(direct, knnArrows)[0][0]

    speed = convert_speed(num1, num2, num3)
    image = cv2.resize(image, (320, 240))

    return image, speed, direct
