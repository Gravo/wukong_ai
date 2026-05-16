# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:10:06 2021

@author: pang
"""

import numpy as np
from grabscreen import grab_screen
import cv2
import time
import directkeys
from getkeys import key_check
import random
from DQN_tensorflow_gpu import DQN
# from DQN_pytorch import DQN
import os
import pandas as pd
from restart import restart
import random
import tensorflow.compat.v1 as tf

def pause_game(paused):
    keys = key_check()
    if 'U' in keys:
        if paused:
            paused = False
            print('start game')
            # time.sleep(1)
        else:
            paused = True
            print('pause game')
            dqn_in_action = False
            # time.sleep(1)
    if paused:
        print('paused')
        while True:
            keys = key_check()
            # pauses game and can get annoying.
            if 'U' in keys:
                if paused:
                    paused = False
                    print('start game')
                    # time.sleep(1)
                    break
                else:
                    paused = True
                    dqn_in_action = False
                    # time.sleep(1)
    return paused


def take_action(action):
    if action == 0:     # n_chooseww
        print("choose go back")
        directkeys.go_back()
        time.sleep(0.1)
        # pass
    elif action == 1:   # j
        print("choose Null")
        # directkeys.go_left()
        time.sleep(0.1)

    elif action == 2:   # k
        print("choose go right")
        directkeys.go_right()
        # directkeys.attack()
        time.sleep(0.1)

    elif action == 3:   # m
        print("choose attack")
        directkeys.attack()
        time.sleep(0.08)
        directkeys.dodge()
        time.sleep(0.02)
        directkeys.attack()
        time.sleep(0.1)

        # directkeys.go_back()
    elif action == 4:   # r
        print("choose dodge")
        directkeys.dodge()
        time.sleep(0.25)
    elif action == 5:   # r
        directkeys.attack()
        time.sleep(0.08)
        directkeys.attack_hit()
        print("choose attack hit")
        time.sleep(0.1)
        # directkeys.go_back()
    # elif action == 6:   # r
    #     directkeys.go_left()
    # elif action == 7:   # r
    #     directkeys.go_right()


def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood, self_courage, next_self_courage, stop, emergence_break):
    # get action reward
    # emergence_break is used to break down training
    # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
    if next_self_blood < 2 and self_blood < 8:     # self dead
        print(f"judge dead, {self_blood},{next_self_blood}")
        if emergence_break < 50:
            reward = -10
            done = 1
            stop = 0
            emergence_break += 1
            return reward, done, stop, emergence_break
        else:
            reward = -10
            done = 1
            stop = 0
            emergence_break = 100
            return reward, done, stop, emergence_break
    elif next_boss_blood - boss_blood > 15:   #boss dead,  not here

        print("Wukong and Tiger should not have")
        reward = 0
        done = 0
        stop = 0

        return reward, done, stop, emergence_break

    elif next_self_courage - self_courage > 5:
        if emergence_break < 50:
            reward = 10
            done = 0
            stop = 0
            # emergence_break += 1
            return reward, done, stop, emergence_break
        else:
            reward = 10
            done = 0
            stop = 0
            emergence_break = 100
            return reward, done, stop, emergence_break

    else:
        self_blood_reward = 0
        boss_blood_reward = 0
        self_courage_reward = 0
        # print(next_self_blood - self_blood)
        # print(next_boss_blood - boss_blood)
        if next_self_blood - self_blood < -10:
            print(f"judge be hit, {self_blood},{next_self_blood}")
            if stop == 0:
                # self_blood_reward = -6
                self_blood_reward = next_self_blood - self_blood
                stop = 1
                # 防止连续取帧时一直计算掉血
        else:
            stop = 0
        if next_boss_blood - boss_blood <= -5:
            print(f"judge hit boss, {boss_blood},{next_boss_blood}")
            boss_blood_reward = boss_blood - next_boss_blood

        if next_self_courage - self_courage >= 5:
            print(f"judge couraged, {self_courage},{next_self_courage}")
            self_courage_reward = next_self_courage - self_courage
        elif next_self_blood - self_blood >= -1:
            #无事发生
            self_blood_reward += 1
        # print("self_blood_reward:    ",self_blood_reward)
        # print("boss_blood_reward:    ",boss_blood_reward)
        reward = self_blood_reward + boss_blood_reward + self_courage_reward
        if boss_blood_reward > 0 and self_blood_reward >= 0 and self_courage_reward >= 0:
            reward = 15
        elif self_blood_reward >= 0 and self_courage_reward >= 0:
            reward = 10
        elif self_blood_reward >= 0 or self_courage_reward >= 0:
            reward = 0
        elif boss_blood_reward > 0:
            reward = 5
        else:
            reward = -5

        done = 0
        emergence_break = 0

        # # 补血会怎样？
        # if next_self_blood < 90 :  # self dead
        #     print(f"judge dead, {self_blood},{next_self_blood}")
        #
        #     import pydirectinput
        #     # pydirectinput.keyDown('2')
        #     pydirectinput.keyDown('R')
        #     time.sleep(0.01)
        #     pydirectinput.keyUp('2')
        #     # pydirectinput.keyUp('R')

        return reward, done, stop, emergence_break
        

DQN_model_path = "model_gpu"
DQN_log_path = "logs_gpu/"
# WIDTH = 96 * 1.5
# HEIGHT = 88 * 1.5
WIDTH = 144
HEIGHT = 132

# window_size = (30, 60, 1000, 600)#384,352  192,176 96,88 48,44 24,22

#wukong training
window_size = (224, 36, 800, 556)  # 384,344  192,172 96,86
# station window_size

blood_window_boss = (390 + 72, 620, 920 - 70, 680 - 25)
blood_window_boss = (462, 647, 848, 655)

def boss_blood_count(boss_gray):
    boss_blood = 0
    for i in range(3, 5): #1,8
        for boss_bd_num in boss_gray[i]:  # the screen height = 35
            # print(boss_bd_num, end = ' ')
        # boss blood gray pixel 90~220
            if boss_bd_num >= 130 and boss_bd_num < 255:
                boss_blood += 1

    print(f'\nboss blood: {boss_blood}')
    return boss_blood



# blood_window_wukong = (115, 660, 270, 740) ##blood not working
# blood_window_wukong = (115, 660, 270, 740) # blood
blood_window_wukong = (145, 690, 273, 690 + 11) # blood


def self_blood_count(self_gray):
    self_blood = 0
    for i in range(4,6): #0,11
        for self_bd_num in self_gray[i]:  ##    for x in range(35,150):
            # print(self_bd_num, end = ' ')
            # self blood gray pixel 80~98
            # 血量灰度值80~98
            if self_bd_num > 120 and self_bd_num <= 255:
                self_blood += 1

    print(f'\nwukong blood: {self_blood}')
    return self_blood




courage_window_wukong = (1100, 660, 1100 + 155, 740)  # qi, 气
courage_window_wukong = (1133, 660, 1133 + 155, 740)  # qi, 气


def self_courage_count(self_gray):
    self_courage = 0
    for i in range(80):
        for self_bd_num in self_gray[i]:  ##    for x in range(35,150):
            # self blood gray pixel 80~98
            # 血量灰度值80~98
            if self_bd_num > 230 and self_bd_num < 255:
                self_courage += 1

    # print("self courage/qi:", self_courage)
    return self_courage

# used to get boss and self bloodv

action_size = 6
# action[n_choose,j,k,m,r]
# j-attack, k-jump, m-defense, r-dodge, n_choose-do nothing

EPISODES = 3000
big_BATCH_SIZE = 16
UPDATE_STEP = 50
# times that evaluate the network
num_step = 0
# used to save log graph
target_step = 0
# used to update target Q network
paused = True
# used to stop training

# global frames
frames = [None] * 60
frame_count = 0  # 记录当前帧的位置
merge_counter = 0  # 合并计数器

if __name__ == '__main__':

    import tensorflow as tf

    # 检查是否有GPU可用
    if tf.config.list_physical_devices('GPU'):
        print("GPU可用")
    else:
        print("GPU不可用")

    from utils import wukong_win_func
    wukong_win_func.set_wukong_position()

    agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path)
    # DQN init
    paused = pause_game(paused)
    # paused at the begin
    emergence_break = 0     
    # emergence_break is used to break down training
    # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
    for episode in range(EPISODES):
        from collections import deque
        # 初始化一个双端队列作为缓存
        frame_cache = deque(maxlen=12)
        t_screen = grab_screen(window_size)
        # 将图像转换为灰度
        screen_gray = cv2.cvtColor(t_screen, cv2.COLOR_BGR2GRAY)

        counter = 0
        # 保存图像
        save_dir = f'runs/screen_shoots_{episode}'
        # save_dir = f'/runs/screen_shoots_{episode}'
        import os
        try:
            # 检查目录是否存在
            if not os.path.exists(save_dir):
                # 如果目录不存在，则创建它
                os.makedirs(save_dir, exist_ok=True)
        except OSError as e:
            print(f"创建目录 {save_dir} 时发生错误: {e}")

        save_path = os.path.join(save_dir, f'screenshot_2x2_{counter}.png')
        cv2.imwrite(save_path, screen_gray)  # 将归一化后的图像恢复原样再保存

        # 将灰度图像添加到缓存中
        # frame_cache.append(screen_gray)
        #
        #
        # screen_gray = cv2.cvtColor(t_img,cv2.COLOR_BGR2GRAY)


        # collect station gray graph
        blood_window_gray_boss = cv2.cvtColor(grab_screen(blood_window_boss),cv2.COLOR_BGR2GRAY)
        blood_window_gray_wukong = cv2.cvtColor(grab_screen(blood_window_wukong), cv2.COLOR_BGR2GRAY)
        courage_window_gray_wukong = cv2.cvtColor(grab_screen(courage_window_wukong), cv2.COLOR_BGR2GRAY)
        # collect blood gray graph for count self and boss blood
        station = cv2.resize(screen_gray,(WIDTH,HEIGHT))
        # change graph to WIDTH * HEIGHT for station input
        boss_blood = boss_blood_count(blood_window_gray_boss)
        self_blood = self_blood_count(blood_window_gray_wukong)
        self_courage = self_courage_count(courage_window_gray_wukong)
        # count init blood
        target_step = 0
        # used to update target Q network
        done = 0
        total_reward = 0
        stop = 0    
        # 用于防止连续帧重复计算reward
        last_time = time.time()
        while True:

            station = np.d(station).reshape(-1,HEIGHT,WIDTH,1)[0]
            # reshape station for tf input placeholder
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            target_step += 1
            # get the action by state
            action = agent.Choose_Action(station)
            take_action(action)
            # take station then the station change
            screen_gray = cv2.cvtColor(grab_screen(window_size),cv2.COLOR_BGR2GRAY)
            # collect station gray graph
            blood_window_boss_gray = cv2.cvtColor(grab_screen(blood_window_boss),cv2.COLOR_BGR2GRAY)
            blood_window_wukong_gray = cv2.cvtColor(grab_screen(blood_window_wukong),cv2.COLOR_BGR2GRAY)
            courage_window_wukong_gray = cv2.cvtColor(grab_screen(courage_window_wukong), cv2.COLOR_BGR2GRAY)
            # collect blood gray graph for count self and boss blood
            next_station = cv2.resize(screen_gray,(WIDTH,HEIGHT))
            next_station = np.array(next_station).reshape(-1,HEIGHT,WIDTH,1)[0]
            next_boss_blood = boss_blood_count(blood_window_boss_gray)
            next_self_blood = self_blood_count(blood_window_wukong_gray)
            next_self_courage = self_courage_count(courage_window_wukong_gray)
            reward, done, stop, emergence_break = action_judge(boss_blood, next_boss_blood,
                                                               self_blood, next_self_blood,
                                                               self_courage, next_self_courage,
                                                               stop, emergence_break)


            # get action rewardv
            if emergence_break == 100:
                # emergence break , save model and paused
                # 遇到紧急情况，保存数据，并且暂停
                print("emergence_break")
                agent.save_model()
                paused = True
            agent.Store_Data(station, action, reward, next_station, done)
            if len(agent.replay_buffer) > big_BATCH_SIZE:
                num_step += 1
                # save loss graph
                # print('train')
                agent.Train_Network(big_BATCH_SIZE, num_step)
            if target_step % UPDATE_STEP == 0:
                agent.Update_Target_Network()
                # update target Q network
            station = next_station
            self_blood = next_self_blood
            boss_blood = next_boss_blood
            total_reward += reward
            paused = pause_game(paused)
            if done == 1:
                break

            # 保存图像
            save_path = os.path.join(save_dir, f'screenshot_2x2_{counter}.png')
            # cv2.imwrite(save_path, screen_gray)  # 将归一化后的图像恢复原样再保存
            # new_img = station
            new_img = screen_gray
            font = cv2.FONT_ITALIC
            font_scale = 0.6
            font_color = (255, 255, 255)  # black
            line_type = 1

            text = f"HP:w {self_blood} , b {boss_blood}; c {self_courage}, a: {action} , r:{reward}"

            # 获取文本大小、位置
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, line_type)
            # text_position = ((new_width - text_width)) // 2, new_height - 20
            # print ("height:",new_height - 20)
            text_position = 2, 15  # ((new_width - text_width)) //

            cv2.putText(new_img, text, text_position, font, font_scale, font_color, line_type)

            cv2.imwrite(save_path, new_img)  # 将归一化后的图像恢复原样再保存
            counter += 1



        if episode % 10 == 0:
            agent.save_model()
            # save model
        print('episode: ', episode, 'Evaluation Average Reward:', total_reward/target_step)
        restart()
