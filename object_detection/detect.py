# This code based on markjay4k code(https://github.com/markjay4k/YOLO-series/)

import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import win32con,win32gui, win32ui
option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.3,
    'gpu':0.5
}

tfnet = TFNet(option)

def grab_screen(winName: str = "Grand Theft Auto V"):
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

#capture = cv2.VideoCapture('gta2.mp4')
t = (0,0,0)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
colors2 = [tuple(t) for i in range(5)]

def light_recog(frame, tl, br):
    roi = frame[tl[1]:br[1], tl[0]:br[0]]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_detected = ''

    red_lower = np.array([136, 87, 111], dtype=np.uint8)
    red_upper = np.array([180, 255, 255], dtype=np.uint8)

    green_lower = np.array([50, 100, 100], dtype=np.uint8)
    green_upper = np.array([70, 255, 255], dtype=np.uint8)

    yellow_lower = np.array([22, 60, 200], np.uint8)
    yellow_upper = np.array([60, 255, 255], np.uint8)

    red = cv2.inRange(hsv, red_lower,red_upper)
    green = cv2.inRange(hsv, green_lower, green_upper)
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    kernel = np.ones((5,5),np.uint8)

    red = cv2.dilate(red, kernel)
    res = cv2.bitwise_and(roi, roi, mask=red)
    green = cv2.dilate(green, kernel)
    res2 = cv2.bitwise_and(roi, roi, mask=green)

    (_,contours,hierarchy) = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in enumerate(contours):
        color_detected = "Red"

    (_,contours,hierarchy) = cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in enumerate(contours):
        color_detected = "Yellow"

    (_,contours,hierarchy) = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in enumerate(contours):
        color_detected = "Green"

    if (0<=tl[1] and br[1] <= 437) and  (153<=tl[0] and br[0] <=630):
        frame = cv2.putText(frame, color_detected, br, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)


def distance_warning(frame, tl, br, confidence):
    mid_x = (br[1] + tl[1]) / 2
    mid_y = (tl[0] + br[0]) / 2
    apx_distance = round((1 - ((br[0] / 800) - (tl[0] / 800))) ** 4, 1)
    frame = cv2.putText(frame, '{}'.format(apx_distance), br, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if apx_distance <= 0.3:
        if confidence >= 0.5:
            if (((mid_x / 800) > 0.3) and ((mid_x / 800) < 0.7)):
                cv2.putText(frame, 'WARNING!!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


while (True):
    stime = time.time()
    screen= np.array(grab_screen("Grand Theft Auto V"), dtype=np.uint8)
    #frame = screen[0:500,0:800]
    results = tfnet.return_predict(screen)
    for color, color2, result in zip(colors, colors2,results):
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        label = result['label']
        confidence = result['confidence']
        text = '{}: {:.0f}%'.format(label, confidence * 100)
        if label == 'traffic light' and confidence > 0.4:
            light_recog(screen,tl,br)
            color = color2
        if (label == 'car' or label == 'bus' or label == 'truck') and tl[1]<490:
            distance_warning(screen,tl,br,confidence)
        screen = cv2.rectangle(screen, tl, br, color, 7)
        screen = cv2.putText(
            screen, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    #screen[0:500,0:800] = frame
    cv2.imshow('frame', screen)
    print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break



