# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 18:31:36 2020

@author: pang
"""

import directkeys
import time

def restart():
    print("死,restart")
    time.sleep(8)
    directkeys.lock_vision()

    time.sleep(30)
    print("开始后跑,restart")
    # from utils import keys_to_boss
    # keys_to_boss.restart_to_tiger()

    from utils import key_listner
    key_listner.replay_keys_from_file("recorded_keys.txt")

    time.sleep(0.4)
    directkeys.attack()
    print("开始新一轮_________")
  
if __name__ == "__main__":  
    restart()