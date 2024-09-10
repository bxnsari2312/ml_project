import tkinter as tk
import customtkinter as ck
import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import cv2
from PIL import Image, ImageTk
from landmarks import landmarks

# Window setup
window = tk.Tk()
window.geometry("480x700")
window.title("Swoleboi")
ck.set_appearance_mode("dark")

# Labels for displaying information
classLabel = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="black", padx=10)
classLabel.place(x=10, y=1)
classLabel.configure(text='STAGE')
counterLabel = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="black", padx=10)
counterLabel.place(x=160, y=1)
counterLabel.configure(text='REPS')
probLabel = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="black", padx=10)
probLabel.place(x=300, y=1)
probLabel.configure(text='PROB')

# Boxes for showing stage, rep count, and probability
classBox = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue")
classBox.place(x=10, y=41)
classBox.configure(text='0')
counterBox = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue")
counterBox.place(x=160, y=41)
counterBox.configure(text='0')
probBox = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue")
probBox.place(x=300, y=41)
probBox.configure(text='0')

# Reset button
def reset_counter():
    global counter
    counter = 0

button = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue")
button.place(x=10, y=600)

# Frame for camera feed
frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)

# Mediapipe pose detection setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Load trained model
with open('deadlift.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)  # 0 for internal camera
current_stage = ''
counter = 0
bodylang_prob = np.array([0, 0])
bodylang_class = ''

# Detection function
def detect():
    global current_stage, counter, bodylang_class, bodylang_prob

    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
        mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))

    try:
        row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        X = pd.DataFrame([row], columns=landmarks)
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0]

        if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "down"
        elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "up"
            counter += 1

    except Exception as e:
        print(e)

    img = image[:, :460, :]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)

    counterBox.configure(text=counter)
    probBox.configure(text=bodylang_prob[bodylang_prob.argmax()])
    classBox.configure(text=current_stage)

detect()
window.mainloop()

