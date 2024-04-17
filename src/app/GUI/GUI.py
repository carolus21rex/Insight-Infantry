import WindowsUtil as wu


def init_gui():
    root = wu.create_gui()
    image_label = wu.add_image(root)
    textbox = wu.add_text_box(root, 4)
    return root, image_label, textbox


def update_gui(image_label, image, textbox, text):
    wu.update_image(image_label, image)
    wu.write_to_textbox(textbox, text)
