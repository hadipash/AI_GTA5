import cv2
import numpy as np
from darkflow.net.build import TFNet
from shapely.geometry import box, Polygon

from data_collection.img_process import grab_screen
from object_detection.direction import Direct

# set YOLO options
options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.3,
    'gpu': 0.5
}
tfnet = TFNet(options)

# capture = cv2.VideoCapture('gta2.mp4')
t = (0, 0, 0)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
colors2 = [tuple(t) for j in range(15)]


def light_recog(frame, direct, traffic_lights):
    traffic_light = traffic_lights[0]

    # find out which traffic light to follow, if there are several
    if len(traffic_lights) > 1:
        # if we need to go to the right
        if direct == Direct.RIGHT or direct == Direct.SLIGHTLY_RIGHT:
            for tl in traffic_lights:
                if tl['topleft']['x'] > traffic_light['topleft']['x']:
                    traffic_light = tl
        # straight or left
        else:
            for tl in traffic_lights:
                if tl['topleft']['x'] < traffic_light['topleft']['x']:
                    traffic_light = tl

    # coordinates of the traffic light
    top_left = (traffic_light['topleft']['x'], traffic_light['topleft']['y'])
    bottom_right = (traffic_light['bottomright']['x'], traffic_light['bottomright']['y'])
    # crop the frame to the traffic light
    roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_detected = ''

    mid_x = (bottom_right[0] + top_left[0]) / 2
    mid_y = (top_left[1] + bottom_right[1]) / 2

    # measure the width of the detected object by asking how many pixels-wide the object is.
    apx_distance = round((1 - ((bottom_right[0] / 800) - (top_left[0] / 800))) ** 18, 1)

    frame = cv2.putText(frame, '{}'.format(apx_distance), (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

    # possible color ranges for traffic lights
    red_lower = np.array([136, 87, 111], dtype=np.uint8)
    red_upper = np.array([180, 255, 255], dtype=np.uint8)

    yellow_lower = np.array([22, 60, 200], dtype=np.uint8)
    yellow_upper = np.array([60, 255, 255], dtype=np.uint8)

    green_lower = np.array([50, 100, 100], dtype=np.uint8)
    green_upper = np.array([70, 255, 255], dtype=np.uint8)

    # find what color the traffic light is showing
    red = cv2.inRange(hsv, red_lower, red_upper)
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green = cv2.inRange(hsv, green_lower, green_upper)

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

    if (0 <= top_left[1] and bottom_right[1] <= 437) and (244 <= top_left[0] and bottom_right[0] <= 630):
        frame = cv2.putText(frame, color_detected, bottom_right, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    frame = cv2.putText(frame, "detected", bottom_right, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    return frame, color_detected, apx_distance


def distance_warning(frame, top_left, bottom_right):
    mid_x = (bottom_right[0] + top_left[0]) / 2
    mid_y = (top_left[1] + bottom_right[1]) / 2
    apx_distance = round((1 - ((bottom_right[0] / 800) - (top_left[0] / 800))) ** 4, 1)
    frame = cv2.putText(frame, '{}'.format(apx_distance), (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

    # myRoi_array= np.array([[(0, 490), (309, 269), (490, 270), (800,473)]])
    # process_img = region_of_interest(frame, myRoi_array)
    # cv2.imshow("precess_img", process_img)

    roi = Polygon([(15, 472), (330, 321), (470, 321), (796, 495)])
    car = box(top_left[0], top_left[1], bottom_right[0], bottom_right[1])

    if roi.intersects(car):
        cv2.putText(frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]],
                    'WARNING!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    return frame, apx_distance


def person_warning(frame, top_left, bottom_right):
    mid_x = (bottom_right[0] + top_left[0]) / 2
    mid_y = (top_left[1] + bottom_right[1]) / 2
    apx_distance = round((1 - ((bottom_right[0] / 800) - (top_left[0] / 800))) ** 15, 1)
    frame = cv2.putText(frame, '{}'.format(apx_distance), (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
    br_array = np.array([[(bottom_right[0], bottom_right[1]), (bottom_right[0], bottom_right[1]),
                          (bottom_right[0], bottom_right[1]), (bottom_right[0], bottom_right[1])]], dtype=np.int32)

    a1 = br_array[0, 0, 0]  # tr[0
    b2 = br_array[0, 0, 1]  # tr[1
    c3 = br_array[0, 1, 0]  # tl[0
    d4 = br_array[0, 1, 1]  # tl[1
    e5 = br_array[0, 2, 0]  # bl[0
    f6 = br_array[0, 2, 1]  # bl[1
    g7 = br_array[0, 3, 0]  # br[0
    h8 = br_array[0, 3, 1]  # br[1

    if apx_distance >= 0.6:
        if a1 <= 629 and b2 >= 360 and c3 >= 179 and d4 >= 37 and e5 >= 0 and f6 <= 503 and g7 <= 800 and h8 <= 500:
            cv2.putText(frame, 'WARNING!', bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    if apx_distance < 0.6:
        if a1 <= 559 and b2 >= 307 and c3 >= 295 and d4 >= 302 and e5 >= 12 and f6 <= 482 and g7 <= 800 and h8 <= 457:
            cv2.putText(frame, 'WARNING!', bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 3)

    return frame


def yolo_detection(screen, direct):
    # find objects on a frame by using YOLO
    results = tfnet.return_predict(screen[:-130, :, :])
    # create a list of detected traffic lights (might be several on a frame)
    traffic_lights = []

    for color, color2, result in zip(colors, colors2, results):
        top_left = (result['topleft']['x'], result['topleft']['y'])
        bottom_right = (result['bottomright']['x'], result['bottomright']['y'])
        label = result['label']
        confidence = result['confidence']
        text = '{}: {:.0f}%'.format(label, confidence * 100)

        if label == 'traffic light' and confidence > 0.3:
            if 220 <= result['topleft']['x'] <= 750:
                traffic_lights.append(result)

            color = color2
            screen = cv2.rectangle(screen, top_left, bottom_right, color, 6)
            screen = cv2.putText(screen, text, top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        if label == 'car' or label == 'bus' or label == 'truck' or label == 'train':
            screen = distance_warning(screen, top_left, bottom_right)
            screen = cv2.rectangle(screen, top_left, bottom_right, color, 6)
            screen = cv2.putText(screen, text, top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        if label == 'person':
            screen = person_warning(screen, top_left, bottom_right)
            screen = cv2.rectangle(screen, top_left, bottom_right, color, 6)
            screen = cv2.putText(screen, text, top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    if traffic_lights:
        screen = light_recog(screen, direct, traffic_lights)

    return screen


def main():
    while True:
        screen = grab_screen()
        screen = yolo_detection(screen, 0)

        cv2.imshow("Frame", screen)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
