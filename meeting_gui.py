import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import load_model
from mtcnn import MTCNN

# Load the models
face_detector = MTCNN()
age_gender_model = load_model("Age_Sex_Detection_Model.keras")

# Initialize the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Age and Gender Detector')
top.configure(background='#cdcdcd')

label1 = Label(top, background='#cdcdcd', font=('arial', 15, 'bold'))
label2 = Label(top, background='#cdcdcd', font=('arial', 15, 'bold'))
sign_image = Label(top)

def detect_and_predict_age_gender(file_path):
    global label1, label2

    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(image_rgb)
    
    if len(faces) < 2:
        label1.configure(text="Error: Upload an image with more than 2 people.")
        label2.configure(text="")
        sign_image.configure(image='')
        return

    age_gender_text = ""
    male_count = 0
    female_count = 0

    for face in faces:
        x, y, width, height = face['box']
        face_img = image_rgb[y:y+height, x:x+width]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = np.expand_dims(face_img, axis=0)
        face_img = face_img / 255.0

        pred = age_gender_model.predict(face_img)
        age = int(np.round(pred[1][0]))
        sex = "Male" if pred[0][0] < 0.5 else "Female"
        
        # Check shirt color
        shirt_color = image_rgb[y+height:y+height+50, x:x+width]
        average_color = np.mean(shirt_color, axis=(0, 1))

        if np.all(average_color > [200, 200, 200]):
            age = 23
        elif np.all(average_color < [50, 50, 50]):
            sex = "Child"
            age = 10
        
        if sex == "Male":
            male_count += 1
        elif sex == "Female":
            female_count += 1
        
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
        cv2.putText(image, f"{sex}, {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        age_gender_text += f"Age: {age}, Gender: {sex}\n"
    
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img_pil)

    sign_image.configure(image=imgtk)
    sign_image.image = imgtk

    label1.configure(text=f"Number of Males: {male_count}, Number of Females: {female_count}")
    label2.configure(text=age_gender_text)

def show_detect_button(file_path):
    detect_btn = Button(top, text="Detect Age and Gender", command=lambda: detect_and_predict_age_gender(file_path), padx=10, pady=5)
    detect_btn.configure(background="#364156", foreground="white", font=("arial", 10, "bold"))
    detect_btn.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label1.configure(text='')
        label2.configure(text='')
        show_detect_button(file_path)
    except Exception as e:
        print(f"Error uploading image: {e}")

upload = Button(top, text='Upload an Image', command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground="white", font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)

label1.pack(side="bottom", expand=True)
label2.pack(side="bottom", expand=True)
heading = Label(top, text="Age and Gender Detector", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#cdcdcd", foreground="#364156")
heading.pack()
top.mainloop()
