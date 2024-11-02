import sys
import cv2
import numpy as np
from keras.models import load_model
import os
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QComboBox, QVBoxLayout, 
                             QPushButton, QFileDialog)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

# Get the current platform (Windows or macOS)
platform = os.name

# Set the folder path based on the platform
if platform == 'nt':  # Windows
    folder_path = os.path.join(os.path.expanduser('~'), 'Documents', 'ingt')
else:  # macOS
    folder_path = os.path.join(os.path.expanduser('~'), 'Documents', 'ingt')

# Load the cascade classifier for face detection
face_cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Load the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer_model_path = os.path.join(folder_path, 'trainer.yml')
if os.path.exists(recognizer_model_path):
    recognizer.read(recognizer_model_path)
else:
    print("Face recognizer model not found.")
    exit()

# Check if the emotion recognition model file exists
emotion_model_path = os.path.join(folder_path, 'emotion_model7.h5')
if os.path.exists(emotion_model_path):
    emotion_model = load_model(emotion_model_path)
else:
    exit()

# Check if the lip moisture detection model file exists
lip_model_path = os.path.join(folder_path, 'wet_dry_model.h5')
if os.path.exists(lip_model_path):
    lip_model = load_model(lip_model_path)
else:
    exit()

def load_stylesheet(filename):
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Файл стилей '{filename}' не найден.")
        return ""
    except Exception as e:
        print(f"Ошибка при загрузке файла стилей: {e}")
        return ""

class FaceEmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Detection, Emotion Recognition, and Lip Moisture Detection")

        # Create layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Create a label for displaying the video
        self.video_label = QLabel()
        self.layout.addWidget(self.video_label)

        # Create a dropdown menu to select the camera
        self.camera_combo = QComboBox()
        self.layout.addWidget(self.camera_combo)

        # Button to start video from camera
        self.videobutton = QPushButton("Start Video")
        self.videobutton.clicked.connect(self.start_video)
        self.layout.addWidget(self.videobutton)

        # Button to open video file
        self.open_video_button = QPushButton("Open Video File")
        self.open_video_button.clicked.connect(self.open_video)
        self.layout.addWidget(self.open_video_button)

        # Create labels to display the results
        self.emotion_label = QLabel("Emotion: ")
        self.layout.addWidget(self.emotion_label)
        self.lip_label = QLabel("Lip Moisture: ")
        self.layout.addWidget(self.lip_label)
        self.stress_level_label = QLabel("Stress Level: ")
        self.layout.addWidget(self.stress_level_label)
        self.threat_probability_label = QLabel("Threat Probability: ")
        self.layout.addWidget(self.threat_probability_label)
        self.prediction_time_label = QLabel("Prediction Time: ")
        self.layout.addWidget(self.prediction_time_label)
        self.accuracy_label = QLabel("Accuracy: ")
        self.layout.addWidget(self.accuracy_label)
        self.recognized_label = QLabel("Recognized ID: ")
        self.layout.addWidget(self.recognized_label)

        # Get a list of available cameras
        self.camera_indices = []
        for i in range(10):  # Try up to 10 cameras
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_indices.append(i)
                cap.release()

        # Populate the camera dropdown
        self.camera_combo.addItems(map(str, self.camera_indices))
        self.camera_combo.currentIndexChanged.connect(self.change_camera)

        # Create a video capture object
        self.cap = cv2.VideoCapture(int(self.camera_combo.currentText()))

        # Set up a timer to update the video frame
        self.timer = QTimer()
        self .timer.timeout.connect(self.update_frame)
        self.timer.start(1)

        self.is_video_file = False  # Flag to determine if a video file is being played

    def open_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Videos (*.mp4 *.avi)")
        if video_path:
            self.cap.release()  # Release the previous video capture
            self.cap = cv2.VideoCapture(video_path)
            self.is_video_file = True  # Set flag that a video file is being played
            self.timer.start(30)  # Start the timer to update the video frame

    def start_video(self):
        self.is_video_file = False  # Установить флаг, что используется камера
        self.cap.release()  # Освобождаем предыдущий видеопоток
        self.cap = cv2.VideoCapture(int(self.camera_combo.currentText()))  # Инициализируем новый видеопоток
        if self.cap.isOpened():
            self.timer.start(30)  # Запускаем таймер для обновления кадров
        else:
            self.emotion_label.setText("Ошибка: Не удалось открыть камеру.")

    def update_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Получаем исходные размеры кадра
                h, w, ch = frame.shape
            
                # Определяем новые размеры с сохранением пропорций
                target_width = 640
                target_height = 480
                aspect_ratio = w / h
            
                if aspect_ratio > 1:  # Широкий формат
                    new_width = target_width
                    new_height = int(target_width / aspect_ratio)
                else:  # Узкий формат
                    new_height = target_height
                    new_width = int(target_height * aspect_ratio)

                # Изменяем размер кадра с сохранением пропорций
                frame = cv2.resize(frame, (new_width, new_height))

                # Обработка для распознавания лиц
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face_roi = gray[y:y + h, x:x + w]
                    face_roi = cv2.resize(face_roi, (48, 48))
                    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
                    face_roi = face_roi / 255.0
                    face_roi = face_roi.reshape((1, 48, 48, 3))

                    start_prediction_time = time.time()
                    emotion_predictions = emotion_model.predict(face_roi)
                    emotion_index = np.argmax(emotion_predictions)
                    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
                    emotion_label = emotion_labels[emotion_index]

                    threat_probability = 0.0
                    if emotion_label in ['Angry', 'Fear', 'Disgust']:
                        threat_probability = 0.7
                    elif emotion_label in ['Sad', 'Surprise']:
                        threat_probability = 0.3
                    else:
                        threat_probability = 0.1

                    stress_level = 0.0
                    if emotion_label in ['Angry', 'Fear', 'Disgust']:
                        stress_level = 0.8
                    elif emotion_label in ['Sad', 'Surprise']:
                        stress_level = 0.5
                    else:
                        stress_level = 0.2

                    lip_predictions = lip_model.predict(face_roi)
                    lip_index = np.argmax(lip_predictions)
                    lip_labels = ['dry', 'wet']
                    lip_label = lip_labels[lip_index]

                    end_prediction_time = time.time()
                    prediction_time = (end_prediction_time - start_prediction_time) * 1000

                    # Face recognition
                    nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
                    self.recognized_label.setText(f"Recognized ID: {nbr_predicted}")

                    self.emotion_label.setText(f"Emotion: {emotion_label}")
                    self.lip_label.setText(f"Lip Moisture: {lip_label}")
                    self.stress_level_label.setText(f"Stress Level: {stress_level:.2f}")
                    self.threat_probability_label.setText(f"Threat Probability: {threat_probability:.2f}")
                    self.prediction_time_label.setText(f"Prediction Time: {prediction_time:.2f} ms")
                    self.accuracy_label.setText(f"Accuracy: {emotion_label}")  # Assuming accuracy is the same as emotion label

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, f"{emotion_label} - {lip_label} - Threat: {threat_probability:.2f} - Stress: {stress_level:.2f} - Prediction Time: {prediction_time:.2f} ms", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_img))
                
            else:
                self.timer.stop()
                self.cap.release()

    def change_camera(self):
        self.cap.release()
        self.cap = cv2.VideoCapture(int(self.camera_combo.currentText()))

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    stylesheet_path = os.path.join(folder_path, 'style.qss')
    stylesheet = load_stylesheet(stylesheet_path)
    if stylesheet:
        app.setStyleSheet(stylesheet)
    window = FaceEmotionApp()
    window.show()
    sys.exit(app.exec_())