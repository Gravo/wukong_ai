"""
Test if pydirectinput can control the game.
Run this while the game is in focus.
"""
import time
import pyautogui
import pydirectinput as pdi

print("Test script starting in 5 seconds...")
print("FOCUS THE GAME WINDOW NOW!")
time.sleep(5)

print("\n[Test 1] Moving mouse right 500px...")
pdi.move(500, 0, relative=True)
time.sleep(1)

print("[Test 2] Moving mouse left 500px...")
pdi.move(-500, 0, relative=True)
time.sleep(1)

print("\n[Test 3] Holding W for 2 seconds...")
pyautogui.keyDown('w')
time.sleep(2)
pyautogui.keyUp('w')

print("\n[Test 4] Turning right (120px) while holding W...")
pyautogui.keyDown('w')
pdi.move(120, 0, relative=True)
time.sleep(0.5)
pyautogui.keyUp('w')

print("\nDone! Did the character move and turn?")
