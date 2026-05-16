# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:14:29 2020

@author: pang
"""
import time

import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

def grab_screen(region=None):

    # print('enter grab screen')
    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img




if __name__ == '__main__':
    #
    # # window_size = (30, 60, 1000, 600)  # 384,352  192,176 96,88 48,44 24,22
    # window_size1 = (47, 520, 246, 690)  # 384,352  192,176 96,88 48,44 24,22
    # window_size1 = (47, 590, 246, 690)  # 384,352  192,176 96,88 48,44 24,22
    # window_size1 = (70, 660, 270, 740)  # 384,352  192,176 96,88 48,44 24,22
    # window_size1 = (115, 660, 270, 740)  # 384,352  192,176 96,88 48,44 24,22
    #
    #
    # window_size2 = (300, 730, 800, 790)  # 384,352  192,176 96,88 48,44 24,22
    # window_size2 = (390, 620, 920, 680)  # 384,352  192,176 96,88 48,44 24,22
    #
    #
    # count = 1000
    # while(True):
    #     time.sleep(5)
    #     screen_gray = cv2.cvtColor(grab_screen(window_size1), cv2.COLOR_BGR2GRAY)
    #     # cv2.imshow("",screen_gray)
    #     file_name = "logs_images/player_" + str(count) + ".jpg"
    #     cv2.imwrite(file_name,screen_gray)
    #     screen_gray = cv2.cvtColor(grab_screen(window_size2), cv2.COLOR_BGR2GRAY)
    #     # cv2.imshow("",screen_gray)
    #     file_name = "logs_images/boss_" + str(count) + ".jpg"
    #     cv2.imwrite(file_name, screen_gray)
    #
    #     print(f"call blood windows, {file_name}")
    #     count += 1



    print("test")


    def self_blood_count(self_gray):
        self_blood = 0
        for self_bd_num in self_gray[469]:
            # self blood gray pixel 80~98
            # 血量灰度值80~98
            if self_bd_num > 90 and self_bd_num < 98:
                self_blood += 1
        return self_blood


    def boss_blood_count(boss_gray):
        boss_blood = 0
        for boss_bd_num in boss_gray[0]:
            # boss blood gray pixel 65~75
            # 血量灰度值65~75
            if boss_bd_num > 65 and boss_bd_num < 75:
                boss_blood += 1
        return boss_blood

    WIDTH = 96
    HEIGHT = 88
    window_size = (30, 60, 1000, 600)  # 384,352  192,176 96,88 48,44 24,22
    # station window_size

    blood_window = (60, 91, 280, 562)
    screen_gray = cv2.cvtColor(grab_screen(window_size), cv2.COLOR_BGR2GRAY)
    blood_window_gray = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_BGR2GRAY)
    cv2.imshow("", blood_window_gray)
    import pygetwindow as gw
    window = gw.getWindowsWithTitle("")[0]

    window.moveTo(1000, 0)


    cv2.waitKey(0)
    # collect blood gray graph for count self and boss blood
    station = cv2.resize(screen_gray, (WIDTH, HEIGHT))
    # change graph to WIDTH * HEIGHT for station input
    boss_blood = boss_blood_count(blood_window_gray)
    self_blood = self_blood_count(blood_window_gray)
    print("blood estimation:",boss_blood,self_blood)
