import cv2
import os
import numpy as np
from PIL import Image
import time

def loadTrainData(fname):
    with np.load(fname) as data:
        train = data['train']
        train_labels = data['train_labels']

    return train, train_labels

def checkDigit(test, train, train_labels):
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    ret, result, neighbours, dist = knn.findNearest(test, k=1)

    return result

def resize11x11(digit_img):
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    ret = cv2.resize(gray, (11,11), fx=1, fy=1, interpolation=cv2.INTER_AREA)
    ret, thr = cv2.threshold(gray, 170, 255,cv2.THRESH_BINARY)

    return thr.reshape(-1, 121).astype(np.float32)

def return_arrow(result):
    if result == 0: return "left"
    elif result ==1: return "right"
    elif result ==2: return "slightly_left"
    elif result ==3: return "slightly_right"
    elif result ==4: return "straight"
    elif result ==5: return "u_turn"
    elif ~(5 >= result >=0): return ""


def resize9x10(digit_img):
  # img = cv2.imread(digit_img)
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    ret = cv2.resize(gray, (9,10), fx=1, fy=1, interpolation=cv2.INTER_AREA)
    ret, thr = cv2.threshold(gray, 170, 255,cv2.THRESH_BINARY)
    cv2.imshow('num',thr)
    cv2.waitKey(0)
    return thr.reshape(-1, 90).astype(np.float32)

def return_num(result):
    if result == 1: return 1
    elif result ==2: return 2
    elif result ==3: return 3
    elif result ==4: return 4
    elif result ==5: return 5
    elif result ==6: return 6
    elif result ==7: return 7
    elif result ==8: return 8
    elif result ==9: return 9
    elif result ==0: return 0
    elif ~(10 > result and result > -1): return ""

def digit_recog(path):
    cov_digit ='digit.npz'
    cov_digit2 ='arrows.npz'

    train, train_labels  = loadTrainData(cov_digit)
    train2, train_labels2  = loadTrainData(cov_digit2)
    start_time = time.time()
    
    savenpz = False
  
    #game screen shot path(only change this part)
    test_img = cv2.imread(path)

    num1 = test_img[572:582,680:689]
    num2 = test_img[572:582,688:697]
    num3 = test_img[572:582,695:704]
    fname = test_img[566:577,17:28]

    #crop the speed & check digit
    test = resize9x10(num1)
    test2 = resize9x10(num2)
    test3 = resize9x10(num3)
    test4 = resize11x11(fname)

    result = checkDigit(test, train, train_labels)
    result2 = checkDigit(test2, train, train_labels)
    result3 = checkDigit(test3, train, train_labels)
    result4 = checkDigit(test4, train2, train_labels2)

    speed=""
    arrows=""
    #speed first digit
    if (result >= 0 and result <= 9):
        speed += str(return_num(result))

    #speed second digit
    if (result2 >= 0 and result2 <= 9):
        speed += str(return_num(result2))
    #speed third digit
    if (result3 >= 0 and result3 <= 9):
        speed += str(return_num(result3))

    if result4 >= 0 and result4 <= 6:
        arrows += return_arrow(result4)
    
    end_time = time.time()

    print (end_time - start_time)
    #speed
    return int(speed) , arrows


#print(digit_recog('C:/Users/pc/Desktop/arrow/captures/test_capture_2.png'))
digit, arrow = digit_recog('C:/Users/pc/Desktop/arrow/captures/test_capture_212.png')
print (digit)
print (arrow)