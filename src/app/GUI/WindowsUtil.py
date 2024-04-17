import tkinter as tk
from PIL import Image, ImageTk


def create_gui():
    root = tk.Tk()
    root.title("EGGIS Deploy Demonstration")
    root.geometry("400x500")
    root.config(bg="black")
    return root


def clear_window(root):
    for widget in root.winfo_children():
        widget.destroy()


def resize(root, x, y):
    root.geometry(f"{x}x{y}")


# used to move from one window format to another
# procedural logic
def change_window(root, module):
    clear_window(root)
    module.entry(root)


def add_label(root, text):
    label_frame = tk.Frame(root, bg="black")
    label_frame.pack(pady=5, padx=10, fill=tk.X)
    label = tk.Label(label_frame, text=text, bg="black", fg="white")
    label.pack(side=tk.LEFT)
    return label


def update_label(root, label, text):
    label.config(text=text)
    root.update_idletasks()


def read_label(label):
    return label.cget("text")


def add_text_box(root, lines):
    textbox = tk.Text(root, height=lines, width=300, fg="white", bg="black")
    textbox.pack(padx=10, pady=(0, 5), anchor="w")
    return textbox


def write_to_textbox(box, text):
    box.delete(1.0, tk.END)
    box.insert(tk.END, text)


# buttons
def add_button(root, text, lamb):
    button = tk.Button(root, text=text, bg="black", fg="white", command=lamb)
    button.pack(pady=5, padx=10, fill=tk.X)
    return button


# sliders
def add_slider(root, label_text, resolution, minimum):
    frame = tk.Frame(root, bg="black")
    frame.pack(fill=tk.X, padx=10, pady=5)

    label = tk.Label(frame, text=label_text, bg="black", fg="white")
    label.pack(side=tk.LEFT)
    slider = tk.Scale(frame, from_=minimum, to=minimum+100*resolution, orient=tk.HORIZONTAL,
                      length=200, resolution=resolution, bg="black", fg="white")
    slider.pack(side=tk.RIGHT)

    slider.bind("<ButtonRelease-1>", lambda event: slider.set(snap_to_resolution(slider.get(), slider)))
    return slider


# snaps sliders to the appropriate position, hard to notice when active, but the sliders look sloppy without it
# functional logic: returns rounded value to the nearest resolution
def snap_to_resolution(value, res):
    min_val = res.cget("from")
    max_val = res.cget("to")
    res = (max_val - min_val) / 100.0
    return round(value/res)*res


def read_slider(slider):
    return slider.get()


def get_image_coordinates(event, imgLocation, image_width, image_height):
    relative_x = event.x / image_width
    relative_y = event.y / image_height
    # print(f'Click at relative coordinates x = {relative_x}, y = {relative_y}')
    imgLocation[0] = relative_x
    imgLocation[1] = relative_y


def add_image(root):
    # create a label and add it to the root
    image_label = tk.Label(root, bg="black")
    image_label.pack(pady=25, fill=tk.BOTH, expand=True)
    return image_label


def update_image(image_label, image):
    resize_image = image.resize((350, 350))

    # convert image for Tkinter
    img = ImageTk.PhotoImage(resize_image)

    # update the image of the passed label
    image_label.config(image=img)
    image_label.image = img  # keep a reference to prevent it from being garbage collected
    return image_label
