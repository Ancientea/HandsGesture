import sys
import cv2
import numpy as np
import os
import tensorflow as tf
from collections import deque
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel,
                            QPushButton, QLineEdit, QVBoxLayout,
                            QWidget, QHBoxLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread


GESTURE_NAMES = ["right_swipe", "left_swipe", "up_swipe", "down_swipe", "click", "pinch"]

class PredictionWorker(QObject):
    finished = pyqtSignal()
    result_ready = pyqtSignal(str)

    def __init__(self, model, sequence):
        super().__init__()
        self.model = model
        self.sequence = sequence

    def predict(self):
        try:
            input_data = np.expand_dims(self.sequence, axis=0)
            pred = self.model.predict(input_data, verbose=0)
            gesture_idx = np.argmax(pred)
            self.result_ready.emit(GESTURE_NAMES[gesture_idx])
        except Exception as e:
            print("预测出错:", e)
            self.result_ready.emit("识别错误")
        self.finished.emit()


class HandGestureApp(QMainWindow):
    update_result_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("手势识别系统")
        self.setGeometry(100, 100, 800, 600)
        self.cap = None  # 初始化cap属性
        self.is_recognizing = False  # 初始化is_recognizing属性
        self.gesture_names = self.load_gesture_names()
        self.init_ui()
        self.init_mediapipe()
        self.init_model()
        self.init_data_buffer()

    def load_gesture_names(self):
        data_dir = "gesture_data"
        return sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    def init_ui(self):
        # 界面组件
        self.camera_id_input = QLineEdit(self)
        self.camera_id_input.setPlaceholderText("输入摄像头 ID")
        self.camera_id_input.setText("0")

        self.start_stop_button = QPushButton("开始识别", self)
        self.start_stop_button.clicked.connect(self.toggle_recognition)

        self.preview_label = QLabel(self)
        self.preview_label.setFixedSize(640, 480)
        self.preview_label.setStyleSheet("background-color: black;")

        self.result_label = QLabel("等待识别...", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 24px; color: red;")

        # 布局
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.camera_id_input)
        control_layout.addWidget(self.start_stop_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.preview_label)
        main_layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 初始化定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.update_result_signal.connect(self.update_result_display)

    def init_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def init_model(self):
        try:
            model_path = "best_model.h5"
            if not os.path.exists(model_path):
                print(f"错误：找不到模型文件 {model_path}")
                self.result_label.setText("错误：找不到模型文件")
                self.model = None
                return
            
            print(f"正在加载模型 {model_path}...")
            print(f"TensorFlow版本: {tf.version.VERSION}")
            print(f"Keras版本: {tf.keras.__version__}")
            
            # 尝试加载模型
            try:
                # 方法1：直接加载
                self.model = tf.keras.models.load_model(model_path, compile=False)
                print("方法1：直接加载成功")
            except Exception as e:
                print(f"方法1失败: {str(e)}")
                try:
                    # 方法2：使用自定义对象加载
                    self.model = tf.keras.models.load_model(
                        model_path,
                        compile=False,
                        custom_objects={
                            'LayerNormalization': tf.keras.layers.LayerNormalization,
                            'Bidirectional': tf.keras.layers.Bidirectional,
                            'LSTM': tf.keras.layers.LSTM
                        }
                    )
                    print("方法2：使用自定义对象加载成功")
                except Exception as e2:
                    print(f"方法2失败: {str(e2)}")
                    try:
                        # 方法3：使用model_from_json和load_weights
                        from tensorflow.keras.models import model_from_json
                        # 加载模型结构
                        with open('model_architecture.json', 'r') as json_file:
                            model_json = json_file.read()
                        self.model = model_from_json(model_json)
                        # 加载权重
                        self.model.load_weights('model_weights.h5')
                        print("方法3：使用model_from_json和load_weights加载成功")
                    except Exception as e3:
                        print(f"方法3失败: {str(e3)}")
                        raise e3
            
            # 编译模型
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("模型编译成功")
            print(f"模型输入形状: {self.model.input_shape}")
            print(f"模型输出形状: {self.model.output_shape}")
            
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            print(f"TensorFlow版本: {tf.version.VERSION}")
            print(f"Keras版本: {tf.keras.__version__}")
            self.result_label.setText("模型加载失败")
            self.model = None

    def init_data_buffer(self):
        self.sequence_length = 30
        self.data_buffer = deque(maxlen=self.sequence_length)
        self.current_gesture = "无动作"
        self.last_prediction = ""

    def toggle_recognition(self):
        if not hasattr(self, 'is_recognizing') or not self.is_recognizing:
            self.start_recognition()
        else:
            self.stop_recognition()

    def start_recognition(self):
        if self.model is None:
            self.result_label.setText("模型未加载")
            return

        camera_id = int(self.camera_id_input.text())
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            self.result_label.setText("摄像头错误")
            return

        self.is_recognizing = True
        self.start_stop_button.setText("停止识别")
        self.timer.start(30)  # ~33fps
        self.data_buffer.clear()
        self.result_label.setText("准备识别...")

    def stop_recognition(self):
        self.is_recognizing = False
        self.start_stop_button.setText("开始识别")
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.preview_label.clear()
        self.result_label.setText("识别已停止")
        self.preview_label.setStyleSheet("background-color: black;")

    def process_landmarks(self, hand_landmarks, frame_width, frame_height):
        # 转换为相对坐标
        root = hand_landmarks.landmark[0]
        frame_data = []
        for lm in hand_landmarks.landmark:
            frame_data.extend([
                lm.x - root.x,  # 相对X坐标
                lm.y - root.y,  # 相对Y坐标
                lm.z - root.z  # 相对Z坐标
            ])
        return frame_data

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # 镜像处理
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 手部检测
        results = self.hands.process(rgb_frame)
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制骨架
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # 处理数据
                frame_data = self.process_landmarks(
                    hand_landmarks,
                    frame.shape[1],
                    frame.shape[0]
                )
                self.data_buffer.append(frame_data)

        # 更新预览画面
        preview_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = preview_frame.shape
        q_img = QImage(preview_frame.data, w, h, w * c, QImage.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(q_img))

        # 当缓冲区满时进行预测
        if len(self.data_buffer) == self.sequence_length:
            self.run_prediction()

        # 未检测到手部时清空缓冲区
        if not hand_detected:
            self.data_buffer.clear()
            self.update_result_signal.emit("无动作")

    def run_prediction(self):
        # 复制当前缓冲区数据
        sequence = np.array(self.data_buffer)

        # 创建预测线程
        self.worker = PredictionWorker(self.model, sequence)
        self.worker_thread = QThread()

        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.predict)
        self.worker.result_ready.connect(self.handle_prediction_result)
        self.worker.finished.connect(self.worker_thread.quit)

        self.worker_thread.start()

    def handle_prediction_result(self, result):
        # 抑制快速变化：仅当连续两次预测相同时更新显示
        if result != self.last_prediction:
            self.last_prediction = result
            self.update_result_signal.emit(result)
        else:
            self.current_gesture = result

    def update_result_display(self, text):
        self.result_label.setText(text)
        # 根据结果改变颜色
        color = "green" if text != "无动作" else "red"
        self.result_label.setStyleSheet(f"font-size: 24px; color: {color};")

    def closeEvent(self, event):
        self.stop_recognition()
        if hasattr(self, 'worker_thread') and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandGestureApp()
    window.show()
    sys.exit(app.exec_())