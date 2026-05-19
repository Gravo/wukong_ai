import ctypes
from ctypes import wintypes, c_long

user32 = ctypes.windll.user32
WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

class RECT(ctypes.Structure):
    _fields_ = [("left", c_long), ("top", c_long), ("right", c_long), ("bottom", c_long)]

def cb(hwnd, l):
    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buf, 256)
    t = buf.value
    if any(k in t.lower() for k in ['黑神话', 'wukong', 'b1']):
        r = RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(r))
        print(f'hwnd={hwnd} title="{t}"')
        print(f'  left={r.left} top={r.top} right={r.right} bottom={r.bottom}')
        print(f'  size={r.right-r.left}x{r.bottom-r.top}')
    return True

user32.EnumWindows(WNDENUMPROC(cb), 0)
