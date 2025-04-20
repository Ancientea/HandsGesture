import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import deque
import datetime
import os
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                            QLabel, QVBoxLayout, QHBoxLayout, QWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# 定义手势类别
GESTURES = ["right_swipe", "left_swipe", "up_swipe", "down_swipe", "click", "pinch"]

class CameraThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def run(self):
        self.cap = cv2.VideoCapture(1)  # 修改为默认摄像头
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.frame_signal.emit(frame)
            else:
                break

    def stop(self):
        self.cap.release()

class MotionAnalyzer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 数据缓冲区
        self.motion_buffer = deque(maxlen=100)
        self.position_buffer = deque(maxlen=100)
        self.is_recording = False
        
        # 统一分析文件配置
        self.analysis_file = "motion_analysis.csv"
        self.init_analysis_file()
        
        # 运动检测参数
        self.MOTION_THRESHOLD = 0.08  # 优化后的运动阈值
        self.STABLE_FRAMES = 5

    def init_analysis_file(self):
        """初始化分析文件"""
        if not os.path.exists(self.analysis_file):
            with open(self.analysis_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'gesture', 'duration_frames', 
                    'max_motion', 'min_motion', 'avg_motion',
                    'start_frame', 'end_frame', 'motion_sequence'
                ])

    def process_landmarks(self, hand_landmarks):
        """处理手部关键点数据（标准化处理）"""
        root = hand_landmarks.landmark[0]
        frame_data = []
        for lm in hand_landmarks.landmark:
            # 标准化到[-1,1]范围
            x = (lm.x - root.x) * 2
            y = (lm.y - root.y) * 2
            z = (lm.z - root.z) * 2
            frame_data.extend([x, y, z])
        return frame_data

    def calculate_motion(self, current_data, prev_data):
        """计算标准化运动量"""
        if prev_data is None:
            return 0
        return np.linalg.norm(np.array(current_data) - np.array(prev_data))

    def analyze_motion(self, frame):
        """分析单帧运动"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 处理数据
            current_data = self.process_landmarks(hand_landmarks)
            prev_data = self.position_buffer[-1] if self.position_buffer else None
            
            # 计算运动量
            motion = self.calculate_motion(current_data, prev_data)
            
            if self.is_recording:
                self.motion_buffer.append(motion)
                self.position_buffer.append(current_data)
            
            # 绘制骨架
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
            
            # 显示实时运动量
            cv2.putText(frame, f"Motion: {motion:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return frame, motion
        return frame, 0

    def save_analysis(self, gesture_name):
        """保存分析结果到CSV"""
        if len(self.motion_buffer) < 10:  # 忽略过短的动作
            return

        with open(self.analysis_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 计算统计指标
            motion_seq = list(self.motion_buffer)
            max_motion = max(motion_seq)
            min_motion = min(motion_seq)
            avg_motion = sum(motion_seq)/len(motion_seq)
            
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                gesture_name,
                len(motion_seq),
                max_motion,
                min_motion,
                avg_motion,
                motion_seq[0],
                motion_seq[-1],
                ';'.join(map("{:.2f}".format, motion_seq))
            ])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = MotionAnalyzer()
        self.camera_thread = CameraThread()
        self.init_ui()
        self.camera_thread.frame_signal.connect(self.update_frame)
        self.camera_thread.start()
        self.current_gesture = None
        self.start_frame = 0

    def init_ui(self):
        self.setWindowTitle("动作分析器")
        self.setGeometry(100, 100, 800, 600)

        # 创建按钮布局
        button_layout = QHBoxLayout()
        self.gesture_buttons = {}
        for gesture in GESTURES:
            btn = QPushButton(gesture)
            btn.clicked.connect(lambda _, g=gesture: self.toggle_recording(g))
            button_layout.addWidget(btn)
            self.gesture_buttons[gesture] = btn

        # 状态显示
        self.status_label = QLabel("准备就绪")
        self.status_label.setAlignment(Qt.AlignCenter)

        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def toggle_recording(self, gesture_name):
        """切换录制状态"""
        if self.analyzer.is_recording:
            self.stop_recording()
        else:
            self.start_recording(gesture_name)

    def start_recording(self, gesture_name):
        self.current_gesture = gesture_name
        self.analyzer.is_recording = True
        self.analyzer.motion_buffer.clear()
        self.analyzer.position_buffer.clear()
        self.status_label.setText(f"正在录制: {gesture_name}")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")

    def stop_recording(self):
        self.analyzer.is_recording = False
        self.analyzer.save_analysis(self.current_gesture)
        self.status_label.setText("录制完成")
        self.status_label.setStyleSheet("color: black; font-weight: normal;")

    def update_frame(self, frame):
        frame = cv2.flip(frame, 1)
        frame, motion = self.analyzer.analyze_motion(frame)
        
        # 自动检测动作边界
        if self.analyzer.is_recording:
            if motion > self.analyzer.MOTION_THRESHOLD and len(self.analyzer.motion_buffer) == 0:
                self.start_frame = len(self.analyzer.motion_buffer)
            
            if motion < self.analyzer.MOTION_THRESHOLD/2 and len(self.analyzer.motion_buffer) > 10:
                self.stop_recording()

        # 转换为QImage并显示
        h, w, c = frame.shape
        bytes_per_line = 3 * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()