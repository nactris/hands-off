import ctypes
import tkinter as tk

root = tk.Tk()
dot_size = 25
root.overrideredirect(True)  
root.attributes("-topmost", True) 
root.attributes("-alpha", 0.6)

try:
  root.wm_attributes("-transparentcolor", "white")
  bg_color = "white"
except tk.TclError:
  bg_color = "systemTransparent"

root.config(bg=bg_color)

canvas = tk.Canvas(
    root,
    width=dot_size,
    height=dot_size,
    bg=bg_color,
    highlightthickness=0,
)
canvas.pack()

canvas.create_oval(
    2, 2, dot_size - 2, dot_size - 2, fill="red", outline="darkred"
)

def get_screen_size():
  w = root.winfo_screenwidth()
  h = root.winfo_screenheight()
  return (w,h)


def move_dot(x, y):
  pos_x = x - (dot_size // 2)
  pos_y = y - (dot_size // 2)
  root.geometry(f"{dot_size}x{dot_size}+{pos_x}+{pos_y}")

def init():
  root.update()  
  hwnd = root.winfo_id()  
  ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
  ctypes.windll.user32.SetWindowLongW(
      hwnd, -20, ex_style | 0x00080000 | 0x00000020
  )
 # root.mainloop()


def tick():
  root.update()