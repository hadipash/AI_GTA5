# This code based on markjay4k code(https://github.com/markjay4k/YOLO-series/)

import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import win32con, win32gui, win32ui
from shapely.geometry import Point, Polygon, LinearRing
from driving.gamepad import AXIS_MIN, AXIS_MAX, TRIGGER_MAX, XInputDevice

'''
option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.3,
    'gpu':0.6
}

tfnet = TFNet(option)
'''

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


# capture = cv2.VideoCapture('gta2.mp4')
t = (0, 0, 0)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
colors2 = [tuple(t) for i in range(15)]


def set_gamepad(controls,gamepad):
    # trigger value
    trigger = int(round(controls[0][1] * TRIGGER_MAX))
    if trigger >= 0:
        # set left trigger to zero
        gamepad.SetTrigger('L', 0)
        gamepad.SetTrigger('R', trigger)
    else:
        # inverse value
        trigger = -trigger
        # set right trigger to zero
        gamepad.SetTrigger('L', trigger)
        gamepad.SetTrigger('R', 0)

    # axis value
    axis = 0
    if controls[0][0] >= 0:
        axis = int(round(controls[0][0] * AXIS_MAX))
    else:
        axis = int(round(controls[0][0] * (-AXIS_MIN)))
    gamepad.SetAxis('X', axis)


def light_recog(frame, tl, br, speed, controls, gamepad, count, direct, direct_array):
    roi = frame[tl[1]:br[1], tl[0]:br[0]]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_detected = ''

    mid_x = (br[0] + tl[0]) / 2
    mid_y = (tl[1] + br[1]) / 2
    apx_distance = round((1 - ((br[0] / 800) - (tl[0] / 800))) ** 18, 1)
    frame = cv2.putText(frame, '{}'.format(apx_distance), (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

    red_lower = np.array([136, 87, 111], dtype=np.uint8)
    red_upper = np.array([180, 255, 255], dtype=np.uint8)

    green_lower = np.array([50, 100, 100], dtype=np.uint8)
    green_upper = np.array([70, 255, 255], dtype=np.uint8)

    yellow_lower = np.array([22, 60, 200], np.uint8)
    yellow_upper = np.array([60, 255, 255], np.uint8)

    red = cv2.inRange(hsv, red_lower, red_upper)
    green = cv2.inRange(hsv, green_lower, green_upper)
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    kernel = np.ones((5, 5), np.uint8)

    red = cv2.dilate(red, kernel)
    res = cv2.bitwise_and(roi, roi, mask=red)
    green = cv2.dilate(green, kernel)
    res2 = cv2.bitwise_and(roi, roi, mask=green)

    (_, contours, hierarchy) = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in enumerate(contours):
        color_detected = "Red"

    (_, contours, hierarchy) = cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in enumerate(contours):
        color_detected = "Yellow"

    (_, contours, hierarchy) = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in enumerate(contours):
        color_detected = "Green"

    if (0 <= tl[1] and br[1] <= 437) and (244 <= tl[0] and br[0] <= 630):
        frame = cv2.putText(frame, color_detected, br, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    num_direct = len(direct_array)

    if num_direct >= 2:
        #straight , left, slightly_left, u_turn
        if (direct == 0 or direct == 1 or direct == 3 or direct ==5):
            print("direct")
            if min(direct_array)==tl[0]:
                frame = cv2.putText(frame, "detected", br, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                if (color_detected == "Red"):
                    if (apx_distance <= 0.7):
                        throttle = -1
                        controls = [[controls[0][0], throttle]]
                        set_gamepad(controls, gamepad)
                        if(speed>=1):
                            throttle = -1
                            set_gamepad(controls, gamepad)
                            time.sleep(0.1)
                    else:
                        throttle = -0.4
                        controls = [[controls[0][0], throttle]]
                        set_gamepad(controls, gamepad)

        #right, slightly_right
        if (direct == 2 or direct == 4):
            if max(direct_array)==tl[0]:
                frame = cv2.putText(frame, "detected", br, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                if (color_detected == "Red"):
                    if (apx_distance <= 0.7):
                        throttle = -1
                        controls = [[controls[0][0], throttle]]
                        set_gamepad(controls, gamepad)
                        if(speed>=1):
                            throttle = -1
                            set_gamepad(controls, gamepad)
                            time.sleep(0.1)
                    else:
                        throttle = -0.4
                        controls = [[controls[0][0], throttle]]
                        set_gamepad(controls, gamepad)

    '''
    0 straight
    1 left
    2 right
    3 slightly_left
    4 slightly_right
    5 u_turn
    6 empty
    '''

def distance_warning(tfnet, frame, tl, br, confidence, controls,  gamepad):
    mid_x = (br[0] + tl[0]) / 2
    mid_y = (tl[1] + br[1]) / 2
    apx_distance = round((1 - ((br[0] / 800) - (tl[0] / 800))) ** 4, 1)
    frame = cv2.putText(frame, '{}'.format(apx_distance), (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

    #myRoi_array= np.array([[(0, 490), (309, 269), (490, 270), (800,473)]])
    #process_img = region_of_interest(frame, myRoi_array)
    #cv2.imshow("precess_img", process_img)

    polygon = Polygon([(15, 472), (330, 321), (470, 321), (796,495)])

    point = Point(br[0],br[1])
    point2 = Point(tl[0],tl[1]+(br[1]-tl[1]))
    if polygon.contains(point) or polygon.contains(point2):
        cv2.putText(frame[tl[1]:br[1], tl[0]:br[0]], 'WARNING!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 3)
        if 0.0<=apx_distance <= 0.4:
            cv2.putText(frame[tl[1]:br[1], tl[0]:br[0]], 'WARNING!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 3)
            throttle = -1
            controls = [[controls[0][0], throttle]]
            set_gamepad(controls, gamepad)
        elif 0.4 < apx_distance <= 0.6:
            cv2.putText(frame[tl[1]:br[1], tl[0]:br[0]], 'WARNING!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 3)
            throttle = -0.4
            controls = [[controls[0][0], throttle]]
            set_gamepad(controls, gamepad)



def person_warning(frame, tl, br, confidence, color, controls, gamepad):
    mid_x = (br[0] + tl[0]) / 2
    mid_y = (tl[1] + br[1]) / 2
    apx_distance = round((1 - ((br[0] / 800) - (tl[0] / 800))) ** 15, 1)
    frame = cv2.putText(frame, '{}'.format(apx_distance), (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
    br_array = np.array([[(br[0], br[1]), (br[0], br[1]), (br[0], br[1]), (br[0], br[1])]], dtype=np.int32)

    a1 = br_array[0, 0, 0]  # tr[0
    b2 = br_array[0, 0, 1]  # tr[1
    c3 = br_array[0, 1, 0]  # tl[0
    d4 = br_array[0, 1, 1]  # tl[1
    e5 = br_array[0, 2, 0]  # bl[0
    f6 = br_array[0, 2, 1]  # bl[1
    g7 = br_array[0, 3, 0]  # br[0
    h8 = br_array[0, 3, 1]  # br[1

    if(apx_distance >=0.6):
        if(a1 <= 629 and b2 >= 360 and c3 >= 179 and d4 >= 37 and e5 >= 0 and f6 <= 503 and g7 <= 800 and h8 <= 500):
            cv2.putText(frame, 'WARNING!', br, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 3)
            throttle = -0.3
            controls = [[controls[0][0], throttle]]
            set_gamepad(controls, gamepad)

    if(apx_distance<0.6):
        if(a1 <= 559 and b2 >= 307 and c3 >= 295 and d4 >= 302 and e5 >= 12 and f6 <= 482 and g7 <= 800 and h8 <= 457):
            cv2.putText(frame, 'WARNING!', br, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 3)
            throttle = -1
            controls = [[controls[0][0], throttle]]
            set_gamepad(controls, gamepad)


def yolo_detection(tfnet, screen, speed, controls, gamepad, direct):
    stime = time.time()
    count = 0

    # frame = screen[0:500,0:800]
    results = tfnet.return_predict(screen[0:500, 0:800])

    direct_array = []
    for item in results:
        data_item = item['label']
        if data_item == "traffic light":
            data_item2 = item['topleft']['x']
            count=count +1
            if 220<=data_item2<=750:
                direct_array.append(data_item2)

    for color, color2, result in zip(colors, colors2, results):
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        label = result['label']
        confidence = result['confidence']
        text = '{}: {:.0f}%'.format(label, confidence * 100)
        if label == 'traffic light' and confidence > 0.3:
            light_recog(screen, tl, br, speed, controls, gamepad, count, direct, direct_array)
            color = color2
            screen = cv2.rectangle(screen, tl, br, color, 6)
            screen = cv2.putText(
                screen, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        if (label == 'car' or label == 'bus' or label == 'truck' or label == 'train') :
            distance_warning(tfnet, screen, tl, br, confidence, controls, gamepad)
            screen = cv2.rectangle(screen, tl, br, color, 6)
            screen = cv2.putText(
                screen, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        if (label == 'person'):
            person_warning(screen,tl, br, confidence, color, controls, gamepad)
            screen = cv2.rectangle(screen, tl, br, color, 6)
            screen = cv2.putText(
                    screen, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
