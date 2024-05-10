import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter
from tkinter.scrolledtext import ScrolledText
from keras.src.utils import img_to_array
from keras.models import model_from_json


class EmotionDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Emotion Detection App")

        self.bg_image = Image.open("background.jpg").filter(ImageFilter.BLUR)
        self.bg_image_tk = ImageTk.PhotoImage(self.bg_image)
        self.bg_label = tk.Label(master, image=self.bg_image_tk)
        self.bg_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.model = model_from_json(open("model3.json", "r").read())
        self.model.load_weights('model3.h5')
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        self.scale_factor = 0.5

        # Create GUI elements with icons
        self.btn_select_image = tk.Button(master, text="Select Image", command=self.select_image)
        self.btn_select_video = tk.Button(master, text="Select Video", command=self.open_camera)
        self.btn_select_folder = tk.Button(master, text="Select Folder", command=self.select_folder)
        self.label_result = tk.Label(master, text="")
        self.btn_about = tk.Button(master, text="About", command=self.show_about)

        self.image_icon = Image.open("image_icon.png").resize((40, 40))
        self.video_icon = Image.open("video_icon.png").resize((40, 40))
        self.folder_icon = Image.open("folder_icon.png").resize((40, 40))
        self.image_icon_tk = ImageTk.PhotoImage(self.image_icon)
        self.video_icon_tk = ImageTk.PhotoImage(self.video_icon)
        self.folder_icon_tk = ImageTk.PhotoImage(self.folder_icon)

        self.btn_select_image.config(image=self.image_icon_tk, compound=tk.LEFT)
        self.btn_select_video.config(image=self.video_icon_tk, compound=tk.LEFT)
        self.btn_select_folder.config(image=self.folder_icon_tk, compound=tk.LEFT)

        self.btn_frame = tk.Frame(master)

        self.btn_frame.pack(expand=True)

        self.btn_select_image.pack(side=tk.TOP, anchor=tk.CENTER, pady=5)
        self.btn_select_video.pack(side=tk.TOP, anchor=tk.CENTER, pady=5)
        self.btn_select_folder.pack(side=tk.TOP, anchor=tk.CENTER, pady=5)
        self.label_result.pack(side=tk.TOP, anchor=tk.CENTER)
        self.btn_about.pack(side=tk.BOTTOM, anchor=tk.SE, padx=10, pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.detect_emotions(file_path)

    def open_camera(self):
        self.detect_emotions(is_video=True)

    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder")
        if folder_path:
            output_file = os.path.join(folder_path, "emotion_predictions.txt")
            with open(output_file, 'w') as file:
                for filename in os.listdir(folder_path):
                    if filename.endswith((".jpg", ".jpeg", ".png")):
                        file_path = os.path.join(folder_path, filename)
                        emotions = self.detect_emotions(file_path)
                        file.write(f'{filename}: {", ".join(emotions)}\n')
            messagebox.showinfo("Success", f"Emotion predictions saved to {output_file}")

    def detect_emotions(self, path=None, is_video=False, is_folder=False):
        if is_video:
            cap = cv2.VideoCapture(0)  # Open the camera
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self.process_frame(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        elif is_folder:
            for file_name in os.listdir(path):
                file_path = os.path.join(path, file_name)
                if file_name.endswith((".jpg", ".jpeg", ".png")):
                    self.detect_emotions(file_path)
        else:
            frame = cv2.imread(path)
            self.process_frame(frame)

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor)
        gray_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image)
        predicted_emotions = []

        for (x, y, w, h) in faces:
            roi_gray = gray_image[y:y + h, x:x + w]
            roi_gray_resized = cv2.resize(roi_gray, (48, 48))
            image_pixels = img_to_array(roi_gray_resized)
            image_pixels = np.expand_dims(image_pixels, axis=0)
            image_pixels /= 255

            predictions = self.model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('angry', 'angry', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            predicted_emotions.append(emotion_prediction)

            cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame_resized, emotion_prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Emotion Detection', frame_resized)
        cv2.waitKey(1)

        if predicted_emotions:
            self.label_result.config(text="Predicted Emotions: " + ", ".join(predicted_emotions))


    def show_about(self):
            about_window = tk.Toplevel(self.master)
            about_window.title("About")

            about_text = """
            Emotion Detection App

            This app allows you to detect emotions in images and real-time 
            video. It uses a pre-trained AI model to recognize facial 
            expressions. This is CNN model, trained on FER-2013 dataset.
            
            FER2013 is a dataset commonly used for training and testing 
            facial expression recognition algorithms. It consists of 
            35,887 grayscale images of faces, each labeled with one 
            of seven emotion categories: anger, disgust, fear, happiness, 
            sadness, surprise, or neutral. The dataset was created by 
            the Facial Expression Recognition 2013 Challenge (FER2013 Challenge) 
            organized by the IEEE Conference on Computer Vision and Pattern 
            Recognition (CVPR) in 2013. It's widely used in the research 
            community for developing and evaluating facial expression 
            recognition models.
            
            This model was trained using Google Collaboratory. It is using 
            5 layers and trained by 100 epochs.
            More information. you can find in my user manual.

            Developed by Oleksii Labiak
            """

            text_widget = tk.Text(about_window, wrap=tk.WORD)
            text_widget.insert(tk.END, about_text)
            text_widget.config(state=tk.DISABLED)
            text_widget.pack(padx=10, pady=10)

            about_window.bind("<Escape>", lambda event: about_window.destroy())

if __name__ == "__main__":
        root = tk.Tk()
        app = EmotionDetectionApp(root)
        root.mainloop()
