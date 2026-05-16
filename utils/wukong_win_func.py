import pygetwindow as gw

def set_wukong_position():

    try:
    # 指定窗口标题
        window_title = "b1  "
        window = gw.getWindowsWithTitle(window_title)[0]
        window.activate()
        window.moveTo(0, 0)
        left, top, right, bottom = window.left, window.top, window.right, window.bottom
        print(left, top, right, bottom)
    except Exception as e:
        print(e)
