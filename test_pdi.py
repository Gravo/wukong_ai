"""
Test if pydirectinput is working correctly.
Run this and check if mouse moves on screen.
"""
import time
import pydirectinput as pdi

print("=" * 60)
print("Testing pydirectinput")
print("1. This will move your WINDOWS MOUSE CURSOR")
print("2. You should see the cursor move on screen")
print("3. If cursor moves but game doesn't respond, game uses Raw Input")
print("=" * 60)
time.sleep(3)

print("\n[Test 1] Moving RIGHT 500px...")
pdi.move(500, 0, relative=True)
time.sleep(1)

print("[Test 2] Moving LEFT 500px...")
pdi.move(-500, 0, relative=True)
time.sleep(1)

print("\n[Test 3] Moving RIGHT 200px (turn_slow)...")
pdi.move(200, 0, relative=True)
time.sleep(0.5)

print("[Test 4] Moving RIGHT 400px (turn_medium)...")
pdi.move(400, 0, relative=True)
time.sleep(0.5)

print("\nDone! Did Windows cursor move?")
print("If YES: pydirectinput works, game uses Raw Input")
print("If NO: pydirectinput not installed or not working")
