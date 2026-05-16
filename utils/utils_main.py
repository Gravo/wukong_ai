import pygetwindow as gw
from PIL import ImageGrab
from datetime import datetime
# from ultralytics import YOLO
import cv2
import numpy as np
import time


def set_wukong_position():

    # 指定窗口标题
    window_title = "b1  "
    window = gw.getWindowsWithTitle(window_title)[0]
    window.activate()
    window.moveTo(0, 0)
    left, top, right, bottom = window.left, window.top, window.right, window.bottom
    print(left, top, right, bottom)


# 窗口， 1280 X 721
def find_boss_blood_location():
    print("boss")
    from PIL import ImageGrab
    # 使用Pillow的ImageGrab来捕获窗口截图
    # window_size2 = (390, 620, 920, 680)  #
    window_size2 = (390 + 72, 620, 920 - 70, 680 - 25)  #
    # screen_gray = cv2.cvtColor(grab_screen(window_size1), cv2.COLOR_BGR2GRAY)
    screenshot = ImageGrab.grab(bbox=window_size2)
    screenshot = screenshot.convert('L')

    #获取图像高度
    boss_hp = estimate_boss_hp(screenshot)

    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H-%M-%S")

    file_name = f"logs/boss_{date_time_str}.png"
    # 保存截图到文件
    # screenshot.save(file_name)

    # 缩放图像
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 定义新的图像尺寸
    width = 300  # 388 * 0.75 = 291
    height = 200 # 35* 0.75 = 26
    resized_frame = cv2.resize(frame, (291, 26))

    #创建新图像
    new_height = resized_frame.shape[0] + 20
    new_width  = resized_frame.shape[1]
    # new_image  = np.zeros((new_height,new_width, 3),dtype= np.uint8)
    new_image = np.zeros((height, width, 3), dtype=np.uint8)

    #将调整图像黏贴
    new_image[height - 26 : height, 0 : 0 + 291] = resized_frame

    #写文本
    bhp = int(boss_hp * 100)
    if bhp >= 1 and bhp <= 100:
        text = f"Boss HP: {bhp}"
    else:
        text = ""
    font  = cv2.FONT_ITALIC
    font_scale = 0.4
    font_color = (255,255,255) #black
    line_type  = 1

    #获取文本大小、位置
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, line_type)
    # text_position = ((new_width - text_width)) // 2, new_height - 20
    # print ("height:",new_height - 20)
    text_position = ((new_width - text_width)) // 2, 15

    cv2.putText(new_image, text, text_position, font, font_scale,font_color, line_type)


    # 显示缩放后的图像
    init = False
    try:
        t1 = "BOSS HP"
        # cv2.imshow(t1, resized_frame)
        cv2.imshow(t1, new_image)

        if init == False:
            # 等待键盘输入，参数是等待时间（毫秒）
            # 如果按下 'q' 键，则退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("exit boss hp")
            w1 = gw.getWindowsWithTitle(t1)[0]
            w1.moveTo(1100, 0)
            # 将窗口设置为最前面
            cv2.setWindowProperty(t1, cv2.WND_PROP_TOPMOST, 1)
            init = True

    except Exception as e:
        print(f"handle boss hp fail：{e}")

    return  bhp


def estimate_boss_hp(screenshot):
    height = screenshot.size[1]
    width = screenshot.size[0]
    hp = 1
    # 遍历
    for x in range(width):
        pixel_value = screenshot.getpixel((x, height - 1))
        if pixel_value >= 90 and  pixel_value <= 220:
            hp += 1
        # print(pixel_value, end=' ')

    est_hp = (hp )/ width
    # print(f"\nsize:{height},{width} boss hp is :",  est_hp)
    return  est_hp


def estimate_wukong_hp_nomral(screenshot):
    screenshot = screenshot.convert('L')
    height = screenshot.size[1]
    width = screenshot.size[0]
    # height = 39
    # width from 35 to 150(should be width - 1)
    width_f = 150 - 35
    hp = 1
    # 遍历
    for x in range(35,150):
        pixel_value = screenshot.getpixel((x, 39))
        if pixel_value >= 90 and  pixel_value <= 220:
            hp += 1
        # print(pixel_value, end=' ')

    est_hp = (hp )/ width_f
    # print(f"\nsize:{height},{width} wukong hp :",  est_hp)
    return  est_hp



def estimate_wukong_hp(screenshot):
    # 将 PIL 图像转换为 NumPy 数组
    image_array = np.array(screenshot)
    self_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    self_blood = 0
    for i in range(40):
        for self_bd_num in self_gray[i]:  ##    for x in range(35,150):
            # self blood gray pixel 80~98
            # 血量灰度值80~98
            print(self_bd_num, end = ' ')
            if self_bd_num > 220 and self_bd_num < 255:
                self_blood += 1

    # print("==============================================")
    # print(self_blood)
    return self_blood


# 窗口， 1280 X 721
def find_wukong_blood_location():
    # print("wukong")
    from PIL import ImageGrab
    # 使用Pillow的ImageGrab来捕获窗口截图

    # window_size1 = (115, 660, 270, 740)  # 384,352  192,176 96,88 48,44 24,22
    window_size1 = (1100, 660, 1100 + 155, 740)  # 384,352  192,176 96,88 48,44 24,22
    # screen_gray = cv2.cvtColor(grab_screen(window_size1), cv2.COLOR_BGR2GRAY)
    screenshot = ImageGrab.grab(bbox=window_size1)


    #获取图像高度
    boss_hp = estimate_wukong_hp(screenshot)

    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H-%M-%S")

    file_name = f"logs/wukong_{date_time_str}.png"
    # 保存截图到文件
    # screenshot.save(file_name)

    # 缩放图像
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 定义新的图像尺寸
    width = 300  # 388 * 0.75 = 291
    height = 200 # 35* 0.75 = 26
    # resized_frame = cv2.resize(frame, (291, 26))
    resized_frame = frame

    #创建新图像
    new_height = resized_frame.shape[0] + 20
    new_width  = resized_frame.shape[1]
    new_image  = np.zeros((height,width, 3),dtype= np.uint8)

    #将调整图像黏贴
    new_image[height - 80 :height, 80:80 + 155] = resized_frame

    #写文本
    bhp = int(boss_hp * 100)
    if bhp >= 1 and bhp <=100:
        text = f"wukong HP: {bhp}"
    else:
        text = ""
    font  = cv2.FONT_ITALIC
    font_scale = 0.4
    font_color = (255,255,255) #black
    line_type  = 1

    #获取文本大小、位置
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, line_type)
    # text_position = ((new_width - text_width)) // 2, new_height - 20
    # print ("height:",new_height - 20)
    text_position = ((new_width - text_width)) // 2, 15

    cv2.putText(new_image, text, text_position, font, font_scale,font_color, line_type)


    # 显示缩放后的图像
    init = False
    try:
        t1 = "Wukong HP"
        # cv2.imshow(t1, resized_frame)
        cv2.imshow(t1, new_image)

        if init == False:
            # 等待键盘输入，参数是等待时间（毫秒）
            # 如果按下 'q' 键，则退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("exit wukong hp")
            w1 = gw.getWindowsWithTitle(t1)[0]
            w1.moveTo(1100, 220)
            # 将窗口设置为最前面
            cv2.setWindowProperty(t1, cv2.WND_PROP_TOPMOST, 1)
            init = True

    except Exception as e:
        print(f"handle wukong hp fail：{e}")

    return  bhp


def check_train_window():
    WIDTH = 96
    HEIGHT = 88
    # window_size = (30, 60, 1000, 600)  # 384,352  192,176 96,88 48,44 24,22
    # window_size = (320, 104, 704, 448)  # 384,344  192,172 96,86
    # window_size = (320 -96, 124 -88, 704 + 96, 468 + 88)  # 384,344  192,172 96,86
    window_size = (224, 36, 800, 556)  # 384,344  192,172 96,86

    screenshot = ImageGrab.grab(bbox=window_size)
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H-%M-%S")
    file_name = f"logs/{date_time_str}_train_window.png"
    # 保存截图到文件
    # screenshot.save(file_name)

    # 缩放图像
    frame = np.array(screenshot)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #灰度
    # 定义新的图像尺寸
    # width = 300  # 388 * 0.75 = 291
    # height = 200  # 35* 0.75 = 26
    height = screenshot.size[1]
    width = screenshot.size[0]
    # print(height,width)
    resized_frame = cv2.resize(frame, ( 98 * 2, 88 *2))
    # resized_frame = frame

    return frame

    # # 显示缩放后的图像
    # init = False
    # try:
    #     t1 = "Train  Windows"
    #     cv2.imshow(t1, resized_frame)
    #
    #     if init == False:
    #         # 等待键盘输入，参数是等待时间（毫秒）
    #         # 如果按下 'q' 键，则退出循环
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             print("exit Train  Windows ")
    #         w1 = gw.getWindowsWithTitle(t1)[0]
    #         w1.moveTo(1000, 320)
    #         # 将窗口设置为最前面
    #         cv2.setWindowProperty(t1, cv2.WND_PROP_TOPMOST, 1)
    #         init = True
    #
    # except Exception as e:
    #     print(f"handle wukong hp fail：{e}")



# 获取窗口的位置和尺寸

if __name__ == '__main__':
    # 假设每个灰度图像是
    WIDTH = 96
    HEIGHT = 88
    import numpy as np

    # 或者初始化为全零图像
    frames = [np.zeros((HEIGHT , WIDTH), dtype=np.uint8) for _ in range(128)]
    set_wukong_position()
    count = 0
    while(True):
        # find_boss_blood_location()
        # find_wukong_blood_location()
        idx = count % 128
        frames[idx] = check_train_window()


        import numpy as np
        import cv2  # 如果你需要显示或保存最终结果


        def merge_images_2x2(frames, idx):
            # 确保 idx 在有效的范围内
            if idx < 0:
                idx += len(frames)  # 如果 idx 是负数，从列表的末尾开始计数

            # 确定要合并的四个图像的索引
            indices = [(idx - i - 1) % len(frames) for i in range(4)]

            # 获取对应的图像
            images_to_merge = [frames[i] for i in indices]

            # 检查所有图像是否具有相同的高度和宽度
            if len(set(image.shape[:2] for image in images_to_merge)) > 1:
                raise ValueError("All images must have the same height and width to merge.")

            # 分别水平拼接两行图像
            top_row = np.hstack((images_to_merge[0], images_to_merge[1]))
            bottom_row = np.hstack((images_to_merge[2], images_to_merge[3]))

            # 垂直拼接两行图像
            final_image = np.vstack((top_row, bottom_row))

            return final_image



        merged_image = merge_images_2x2(frames, idx)

        # 可视化结果
        cv2.imshow('Merged Image', frames[idx] )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # now = datetime.now()
        # date_time_str = now.strftime("%Y%m%d_%H-%M-%S")
        # file_name = f"logs/{date_time_str}_merge_windows.png"
        # # 保存截图到文件
        # merged_image.save(file_name)

        count += 1
        time.sleep(0.05)



