import cv2
import numpy as np
import win32gui, win32ui, win32con

# name of the window to grab
winName = "Untitled - Notepad"


def grab_screen(region=None):
    desktop = win32gui.GetDesktopWindow()

    while True:
        # screen capture region
        if region:
            left, top, x2, y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
        else:
            gtawin = win32gui.FindWindow(None, winName)
            # get the bounding box of the window
            left, top, x2, y2 = win32gui.GetWindowRect(gtawin)
            top += 40  # for title bar
            width = x2 - left + 1
            height = y2 - top + 1

        hwindc = win32gui.GetWindowDC(desktop)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(desktop, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        cv2.imshow('test', cv2.cvtColor(img, cv2.COLOR_BGRA2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


if __name__ == "__main__":
    grab_screen()
