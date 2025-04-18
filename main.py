import sys
import cv2
import numpy as np
import os
import tensorflow as tf
from collections import deque
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel,
                            QPushButton, QLineEdit, QVBoxLayout,
                            QWidget, QHBoxLayout, QMessageBox)
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
        self.is_running = False

    def predict(self):
        self.is_running = True
        try:
            input_data = np.expand_dims(self.sequence, axis=0)
            pred = self.model.predict(input_data, verbose=0)
            gesture_idx = np.argmax(pred)
            if self.is_running:  # 检查是否应该继续
                self.result_ready.emit(GESTURE_NAMES[gesture_idx])
        except Exception as e:
            print("预测出错:", e)
            if self.is_running:  # 检查是否应该继续
                self.result_ready.emit("识别错误")
        finally:
            self.is_running = False
            self.finished.emit()

    def stop(self):
        self.is_running = False


class HandGestureApp(QMainWindow):
    update_result_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("手势识别系统")
        self.setGeometry(100, 100, 800, 600)
        self.cap = None  # 初始化cap属性
        self.is_recognizing = False  # 初始化is_recognizing属性
        self.worker = None
        self.worker_thread = None
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

        self.test_button = QPushButton("测试模型", self)
        self.test_button.clicked.connect(self.test_model)

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
        control_layout.addWidget(self.test_button)

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
            min_tracking_confidence=0.5,
            model_complexity=0  # 使用较轻量级的模型
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def init_model(self):
        try:
            model_path = "saved_model"
            if not os.path.exists(model_path):
                print(f"错误：找不到模型文件夹 {model_path}")
                self.result_label.setText("错误：找不到模型文件夹")
                self.model = None
                return
            
            print(f"正在加载模型 {model_path}...")
            print(f"TensorFlow版本: {tf.version.VERSION}")
            print(f"Keras版本: {tf.keras.__version__}")
            
            # 加载SavedModel格式的模型
            try:
                self.model = tf.keras.models.load_model(model_path)
                print("模型加载成功")
            except Exception as e:
                print(f"模型加载失败: {str(e)}")
                self.result_label.setText("模型加载失败")
                self.model = None
                return
            
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
        
        # 停止预测线程
        if self.worker is not None:
            self.worker.stop()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            if not self.worker_thread.wait(1000):
                self.worker_thread.terminate()
            self.worker_thread = None
        self.worker = None
            
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
        
        # 获取图像尺寸
        image_height, image_width, _ = frame.shape
        
        # 转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 手部检测
        results = self.hands.process(rgb_frame)
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制骨架
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                # 处理数据
                frame_data = self.process_landmarks(
                    hand_landmarks,
                    image_width,
                    image_height
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
        try:
            # 如果已有正在运行的线程，先停止它
            if self.worker is not None:
                self.worker.stop()
            if self.worker_thread is not None and self.worker_thread.isRunning():
                self.worker_thread.quit()
                self.worker_thread.wait(1000)  # 等待最多1秒

            # 创建新的预测线程
            sequence = np.array(self.data_buffer)
            self.worker = PredictionWorker(self.model, sequence)
            self.worker_thread = QThread()

            # 设置连接
            self.worker.moveToThread(self.worker_thread)
            self.worker_thread.started.connect(self.worker.predict)
            self.worker.result_ready.connect(self.handle_prediction_result)
            self.worker.finished.connect(lambda: self.cleanup_thread())

            # 启动线程
            self.worker_thread.start()
        except Exception as e:
            print(f"创建预测线程时出错: {str(e)}")

    def cleanup_thread(self):
        try:
            if self.worker_thread is not None:
                self.worker_thread.quit()
                if not self.worker_thread.wait(1000):
                    self.worker_thread.terminate()
                self.worker_thread = None
            self.worker = None
        except Exception as e:
            print(f"清理线程时出错: {str(e)}")

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

    def test_model(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "模型未加载，请先加载模型")
            return

        try:
            # 加载测试数据
            test_data = np.load("test_data.npy")
            if test_data.shape[0] < 30:
                QMessageBox.warning(self, "警告", "测试数据不足30帧")
                return

            # 进行预测
            input_data = np.expand_dims(test_data, axis=0)
            pred = self.model.predict(input_data, verbose=0)
            gesture_idx = np.argmax(pred)
            confidence = pred[0][gesture_idx]
            
            result = f"预测结果: {GESTURE_NAMES[gesture_idx]}\n置信度: {confidence:.2%}"
            QMessageBox.information(self, "测试结果", result)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"测试过程中出错: {str(e)}")

    def closeEvent(self, event):
        self.stop_recognition()
        event.accept()

    def __del__(self):
        self.stop_recognition()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandGestureApp()
    window.show()
    sys.exit(app.exec_())