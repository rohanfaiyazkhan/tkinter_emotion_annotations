from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from pathlib import Path
from utils import verify_image, resize_with_padding
import os
import datetime

root_dir = Path("./sampled_train_set")
image_dir = root_dir / "images"
annotations_dir = root_dir / "annotations"

save_state_dir = Path("./save_state")
labels_file = save_state_dir / "labels.csv"

image_paths = sorted(os.listdir(image_dir))
total_num_of_images = len(image_paths)

emotion_labels = {
    0: "Neutral",
    1: "Happiness",
    2: "Sadness",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt"
}

actor_code = {
    0: "Not Actor",
    1: "Actor"
}

public_figure = {
    0: "Not Public Figure",
    1: "Public Figure"
}

posing_code = {
    0: "Not Posing",
    1: "Posing"
}

looking_code = {
    0: "Not Looking At Camera",
    1: "Looking At Camera"
}

label_makes_sense = {
    0: "Makes sense",
    1: "Does not make sense"
}

facial_expression_codes = {
    0: "Neutral",
    1: "Scowl",
    2: "Frown",
    3: "Smile",
    6: "Other"
}

skin_tone_colors = {
    0: "#BBA78E",
    1: "#AD9073",
    2: "#9C7E63",
    3: "#875F42",
    4: "#764A25",
    5: "#341F1C"
}

code_classes = {
    "actor": {"text": "Is this an actor?", "codes": actor_code},
    "public_figure": {"text": "If not an actor, is this a public figure?", "codes": public_figure},
    "posing": {"text": "Is this person posing?", "codes": posing_code},
    "looking_at_camera": {"text": "Is this person looking straight at the camera?", "codes": looking_code},
    "label_makes_sense": {"text": "Does the label make any sense?", "codes": label_makes_sense},
    "facial_expression": {"text": "What is the facial expression?", "codes": facial_expression_codes},
    "skin_tone": {"text": "Which color matches person's skin tone?", "codes": skin_tone_colors}
}

data_columns = ('img_filename', 'original_emotion_label', 'actor', 'public_figure',
                'posing', 'looking_at_camera', 'label_makes_sense', 'facial_expression', 'skin_tone')
binary_label_columns = ('actor', 'public_figure',
                        'posing', 'looking_at_camera', 'label_makes_sense')


def get_dataframe():
    if labels_file.exists():
        return pd.read_csv(labels_file)
    else:
        df = pd.DataFrame(columns=data_columns)
        df['img_filename'] = image_paths
        return df


def get_first_index_that_was_not_set(labels_df):
    relevant_rows = ['original_emotion_label', 'actor', 'public_figure',
                     'posing', 'looking_at_camera', 'label_makes_sense', 'facial_expression']
    relevant_df = labels_df[relevant_rows]
    for idx, row in relevant_df.iterrows():
        if row.isnull().values.any():
            return idx

    return -1


class Window(Frame):
    # tuples with [actor_code, posing_code, looking_code, agree_with_label]
    # -1 = unlabeled

    prev_next_btns = None
    save_button = None
    label_pickers = [None, None, None, None, None, None]
    labels_df = None

    def __init__(self, master, initial_index=0):
        Frame.__init__(self, master)

        self.master = master
        self.pack(fill=BOTH, expand=1)
        self.index = initial_index

        self.labels_df = get_dataframe()

        first_index = get_first_index_that_was_not_set(
            self.labels_df)

        if first_index > -1:
            self.index = first_index
            self.load_image(first_index)

        self.load_buttons()
        self.load_label_pickers()
        self.load_facial_expression_picker()
        self.load_skin_tone_picker()

    def save(self):
        self.labels_df.to_csv(labels_file)

        print(f"[{datetime.datetime.now()}]: Successfully saved")

    def load_image(self, index):
        img_fn = self.labels_df.at[index, 'img_filename']
        img_path = image_dir / img_fn

        if verify_image(img_path):
            file_num = img_path.stem
            annotation_file = annotations_dir / f'{file_num}_exp.npy'
            label = int(np.load(annotation_file))

            if label not in emotion_labels:
                raise Exception(
                    f"Emotion label not in label dictionary: {label}")

            self.labels_df.at[index, 'original_emotion_label'] = label
            self.label_text = emotion_labels[label]

            if hasattr(self, "file_label"):
                self.file_label.grid_forget()

            self.file_label = Label(
                self, text=f'Viewing: {img_fn} ({index + 1}/{total_num_of_images}) Emotion: {self.label_text}')

            self.file_label.grid(row=0, column=0, columnspan=5)

            if hasattr(self, "img"):
                self.img.grid_forget()

            load = Image.open(img_path)
            resized = resize_with_padding(load)
            self._render = ImageTk.PhotoImage(resized)

            self.img = Label(self, image=self._render)
            self.img.grid(row=1, column=0, columnspan=5)
        else:
            error = Label(self, text=f'Image: {img_path} could not be opened')

            error.grid_forget()
            error.grid(row=1, column=0, columnspan=5)

    def next_handler(self):
        self.index += 1

        self.load_image(self.index)
        self.load_buttons()
        self.load_label_pickers()
        self.load_facial_expression_picker()
        self.load_skin_tone_picker()

    def prev_handler(self):
        self.index -= 1

        self.load_image(self.index)
        self.load_buttons()
        self.load_label_pickers()
        self.load_facial_expression_picker()
        self.load_skin_tone_picker()

    def get_label_change_handler(self, curr_index, col_name):
        def command(*args):
            current_label = self.labels_df.at[curr_index, col_name]
            self.labels_df.at[curr_index,
                              col_name] = 0 if current_label == 1 else 1
            self.load_label_pickers()

        return command

    def get_facial_exp_change_handler(self, current_index, code):
        def command(*args):
            self.labels_df.at[current_index, "facial_expression"] = code
            self.load_facial_expression_picker()

        return command

    def get_skin_tone_change_handler(self, current_index, code):
        def command(*args):
            self.labels_df.at[current_index, "skin_tone"] = code
            self.load_skin_tone_picker()

        return command

    def load_label_pickers(self):
        current_index = self.index

        if hasattr(self, "label_text_display"):
            self.label_text_display.grid_forget()

        self.label_text_display = Label(
            self, text=f"Predicted label: {self.label_text}")
        self.label_text_display.grid(row=2, column=0, columnspan=6)

        labels = self.labels_df.iloc[current_index]

        for colname in binary_label_columns:
            if (np.isnan(labels[colname])):
                self.labels_df.at[current_index, colname] = 0
                labels[colname] = 0

        for i, name in enumerate(binary_label_columns):

            code_dictionary = code_classes[name]
            text = code_dictionary["text"]

            frame = Frame(self, bd=6)
            instruction = Label(frame, text=text)

            if labels[name] == 0:
                btn_bg, btn_fg = "white", "black"
            else:
                btn_bg, btn_fg = "#525252", "white"

            btn_text = code_dictionary["codes"][labels[name]]

            button = Button(frame, command=self.get_label_change_handler(
                current_index, name), bg=btn_bg, fg=btn_fg, text=btn_text)

            instruction.pack()
            button.pack()

            if self.label_pickers[i] != None:
                self.label_pickers[i].grid_forget()

            frame.grid(row=3, column=i)
            self.label_pickers[i] = frame

    def load_facial_expression_picker(self):
        current_index = self.index
        labels = self.labels_df.iloc[current_index]
        fe_label = labels["facial_expression"]

        if np.isnan(fe_label):
            self.labels_df.at[current_index, "facial_expression"] = 0
            fe_label = 0

        self.facial_exp_frame = Frame(self, bd=6)
        label = Label(
            self.facial_exp_frame, text=code_classes["facial_expression"]["text"])
        label.grid(
            row=0, column=-0, columnspan=len(facial_expression_codes))

        for i, exp in facial_expression_codes.items():
            if fe_label == i:
                btn_bg, btn_fg = "#525252", "white"
            else:
                btn_bg, btn_fg = "white", "black"

            button = Button(self.facial_exp_frame, text=exp, bg=btn_bg,
                            fg=btn_fg, command=self.get_facial_exp_change_handler(current_index, i))
            button.grid(column=i, row=1, padx=6)

        self.facial_exp_frame.grid(row=4, column=0, columnspan=5)

    def load_skin_tone_picker(self):
        current_index = self.index
        labels = self.labels_df.iloc[current_index]
        sk_label = labels["skin_tone"]

        if np.isnan(sk_label):
            self.labels_df.at[current_index, "skin_tone"] = 0
            sk_label = 0

        self.skin_tone_frame = Frame(self, bd=6)
        label = Label(
            self.skin_tone_frame, text=code_classes["skin_tone"]["text"])
        label.grid(
            row=0, column=-0, columnspan=len(skin_tone_colors))

        for i, color in skin_tone_colors.items():
            if sk_label == i:
                text = u'\u2713'
            else:
                text = ''

            button = Button(self.skin_tone_frame, text=text, bg=color,
                            fg="white", width=6, height=3, command=self.get_skin_tone_change_handler(current_index, i))
            button.grid(column=i, row=1, padx=6)

        self.skin_tone_frame.grid(row=5, column=0, columnspan=5)

    def load_buttons(self):
        if self.prev_next_btns != None:
            self.prev_next_btns.grid_forget()

        self.prev_next_btns = Frame(self)

        prev_btn = Button(self.prev_next_btns,
                          text="Previous")
        next_btn = Button(self.prev_next_btns, text="Next")

        prev_status = DISABLED if self.index <= 0 else ACTIVE
        next_status = DISABLED if self.index > total_num_of_images - 1 else ACTIVE

        prev_btn.config(
            text="Previous", state=prev_status, command=self.prev_handler)
        next_btn.config(text="Next", state=next_status,
                             command=self.next_handler)

        prev_btn.grid(row=0, column=0, padx=8)
        next_btn.grid(row=0, column=1, padx=8)
        self.prev_next_btns.grid(row=6, column=1, columnspan=3)

        if self.save_button == None:
            self.save_button = Button(self, text="Save")

        self.save_button.config(command=self.save)
        self.save_button.grid_forget()
        self.save_button.grid(row=6, column=4, pady=8)


root = Tk()
root.geometry("1080x768")
root.wm_title("Label Images")

app = Window(root)


def on_closing():
    try:
        if messagebox.askyesno("Quit", "Do you want to save before closing?"):
            app.save()
            root.destroy()
        else:
            print(f"[{datetime.datetime.now()}]: Exited without saving")
            root.destroy()
    except Exception as e:
        print(e)
        root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
