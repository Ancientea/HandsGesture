import sys
import os
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout
import mediapipe as mp

# 定义手势类别
GESTURES = ["right_swipe", "left_swipe", "up_swipe", "down_swipe", "click", "pinch"]
DATA_DIR = "gesture_data"  # 数据集保存目录
FRAME_COUNT = 30  # 1秒采集30帧
mp_drawing = mp.solutions.drawing_utils  # 添加绘图工具


class CameraThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.frame_signal.emit(frame)
            else:
                break

    def stop(self):
        self.cap.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_mediapipe()
        self.current_gesture = None
        self.frames_buffer = []
        self.camera_thread = CameraThread()
        self.camera_thread.frame_signal.connect(self.update_frame)
        self.camera_thread.start()

    def init_ui(self):
        self.setWindowTitle("手势数据收集")
        self.setGeometry(100, 100, 800, 600)

        # 创建按钮布局
        button_layout = QHBoxLayout()
        self.gesture_buttons = {}
        for gesture in GESTURES:
            btn = QPushButton(gesture)
            btn.clicked.connect(lambda _, g=gesture: self.start_collect(g))
            button_layout.addWidget(btn)
            self.gesture_buttons[gesture] = btn

        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def init_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )

    def start_collect(self, gesture):
        self.current_gesture = gesture
        self.frames_buffer = []
        print(f"开始收集手势: {gesture}")

    def process_frame(self, frame):
        # 转换为RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # 初始化坐标列表
        original_landmarks = []
        mirrored_landmarks = []
        draw_frame = None  # 用于绘制骨架的帧

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # 绘制骨架到原始帧
            draw_frame = rgb_frame.copy()
            mp_drawing.draw_landmarks(
                draw_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

            # 计算相对坐标
            root = hand_landmarks.landmark[0]
            for lm in hand_landmarks.landmark:
                # 原始相对坐标
                dx = lm.x - root.x
                dy = lm.y - root.y
                dz = lm.z - root.z
                original_landmarks.extend([dx, dy, dz])

                # 镜像相对坐标（x方向取反）
                mirrored_landmarks.extend([-dx, dy, dz])
        else:
            original_landmarks = [0] * 63
            mirrored_landmarks = [0] * 63

        return original_landmarks, mirrored_landmarks, draw_frame

    def update_frame(self, frame):
        # 处理原始帧并获取坐标
        orig_landmarks, mirror_landmarks, draw_frame = self.process_frame(frame)

        # 显示带骨架的镜像画面
        if draw_frame is not None:
            # 镜像处理显示画面
            mirrored_display = cv2.flip(draw_frame, 1)
        else:
            # 显示普通镜像画面
            mirrored_display = cv2.flip(frame, 1)

        # 绘制状态圆圈
        h, w, _ = mirrored_display.shape
        circle_color = (0, 255, 0) if self.current_gesture and (orig_landmarks != [0] * 63) else (0, 0, 255)
        cv2.circle(mirrored_display, (50, 50), 20, circle_color, -1)

        # 转换为 QImage 并显示
        q_img = QImage(mirrored_display.data, w, h, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

        # 如果正在收集数据且检测到手部
        if self.current_gesture and (orig_landmarks != [0] * 63):
            self.frames_buffer.append((orig_landmarks, mirror_landmarks))

            # 收集满30帧后保存
            if len(self.frames_buffer) >= FRAME_COUNT:
                self.save_data()
                self.current_gesture = None

    def save_data(self):
        # 保存原始数据和镜像数据
        original_data = [f[0] for f in self.frames_buffer]
        mirrored_data = [f[1] for f in self.frames_buffer]

        # 根据手势类型保存数据
        if self.current_gesture in ["right_swipe", "left_swipe"]:
            # 保存镜像数据到当前标签
            self.save_single_gesture(mirrored_data, self.current_gesture)

            # 保存原始数据到相反标签
            opposite_gesture = "left_swipe" if self.current_gesture == "right_swipe" else "right_swipe"
            self.save_single_gesture(original_data, opposite_gesture)
        else:
            # 保存两份数据到当前标签
            self.save_single_gesture(original_data, self.current_gesture)
            self.save_single_gesture(mirrored_data, self.current_gesture)

        print(f"已保存{len(self.frames_buffer)}帧数据")

    def save_single_gesture(self, data, gesture):
        save_dir = os.path.join(DATA_DIR, gesture)
        os.makedirs(save_dir, exist_ok=True)

        # 生成唯一文件名
        file_count = len(os.listdir(save_dir))
        save_path = os.path.join(save_dir, f"{file_count}.npy")

        # 保存为numpy数组 (30帧 × 63特征)
        np.save(save_path, np.array(data))
        print(f"保存数据到: {save_path}")

    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())