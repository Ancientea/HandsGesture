import sys
import os
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout
import mediapipe as mp
from collections import deque

# 定义手势类别
GESTURES = ["right_swipe", "left_swipe", "up_swipe", "down_swipe", "click", "pinch"]
DATA_DIR = "gesture_data"  # 数据集保存目录
FRAME_COUNT = 30  # 1秒采集30帧
mp_drawing = mp.solutions.drawing_utils  # 添加绘图工具

# 动作检测参数
MOTION_THRESHOLD = 0.08  # 与test.py保持一致
STABLE_FRAMES = 5  # 稳定帧数阈值

class CameraThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def run(self):
        self.cap = cv2.VideoCapture(1)
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
        self.motion_buffer = deque(maxlen=100)  # 用于存储运动量
        self.position_buffer = deque(maxlen=100)  # 用于存储位置数据
        self.stable_frames = 0  # 稳定帧计数
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
        self.motion_buffer.clear()
        self.position_buffer.clear()
        self.stable_frames = 0
        self.status_label.setText(f"正在收集: {gesture}")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        print(f"开始收集手势: {gesture}")

    def process_frame(self, frame):
        # 转换为RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # 初始化坐标列表
        original_landmarks = []
        mirrored_landmarks = []
        draw_frame = None  # 用于绘制骨架的帧
        motion = 0  # 初始化运动量

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # 绘制骨架到原始帧
            draw_frame = rgb_frame.copy()
            mp_drawing.draw_landmarks(
                draw_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

            # 计算标准化坐标
            root = hand_landmarks.landmark[0]
            current_data = []
            for lm in hand_landmarks.landmark:
                # 标准化到[-1,1]范围
                x = (lm.x - root.x) * 2
                y = (lm.y - root.y) * 2
                z = (lm.z - root.z) * 2
                current_data.extend([x, y, z])
                
                # 原始相对坐标
                dx = lm.x - root.x
                dy = lm.y - root.y
                dz = lm.z - root.z
                original_landmarks.extend([dx, dy, dz])

                # 镜像相对坐标
                mirrored_landmarks.extend([-dx, dy, dz])

            # 计算运动量
            if self.position_buffer:
                motion = np.linalg.norm(np.array(current_data) - np.array(self.position_buffer[-1]))
            
            # 更新缓冲区
            self.position_buffer.append(current_data)
            self.motion_buffer.append(motion)

            # 在画面上显示运动量
            cv2.putText(draw_frame, f"Motion: {motion:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            original_landmarks = [0] * 63
            mirrored_landmarks = [0] * 63

        return original_landmarks, mirrored_landmarks, draw_frame, motion

    def update_frame(self, frame):
        # 处理原始帧并获取坐标
        orig_landmarks, mirror_landmarks, draw_frame, motion = self.process_frame(frame)

        # 显示带骨架的镜像画面
        if draw_frame is not None:
            mirrored_display = cv2.flip(draw_frame, 1)
        else:
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
            # 动作检测逻辑
            if motion > MOTION_THRESHOLD:
                self.stable_frames = 0
                if len(self.frames_buffer) == 0:
                    print(f"检测到动作开始 (运动量: {motion:.4f})")
                self.frames_buffer.append((orig_landmarks, mirror_landmarks))
            else:
                self.stable_frames += 1
                if len(self.frames_buffer) > 0:
                    self.frames_buffer.append((orig_landmarks, mirror_landmarks))

            # 检查是否需要保存数据
            if len(self.frames_buffer) >= FRAME_COUNT or (len(self.frames_buffer) > 10 and self.stable_frames >= STABLE_FRAMES):
                if self.validate_data_quality():
                    self.save_data()
                else:
                    print("数据质量不合格，请重新执行")
                self.current_gesture = None
                self.frames_buffer = []
                self.stable_frames = 0
                self.status_label.setText("准备就绪")
                self.status_label.setStyleSheet("color: black; font-weight: normal;")

    def validate_data_quality(self):
        """验证数据质量"""
        if len(self.frames_buffer) < 10:  # 最少需要10帧
            print("警告：帧数不足")
            return False

        # 计算运动特征
        motions = list(self.motion_buffer)
        max_motion = max(motions)
        avg_motion = sum(motions) / len(motions)
        
        # 根据手势类型设置不同的阈值
        if self.current_gesture in ["right_swipe", "left_swipe"]:
            # 滑动类手势需要较大的运动量
            if max_motion < 0.1 or avg_motion < 0.05:
                print(f"警告：{self.current_gesture}动作幅度过小 (最大: {max_motion:.4f}, 平均: {avg_motion:.4f})")
                return False
        elif self.current_gesture in ["up_swipe", "down_swipe"]:
            # 上下滑动需要适中的运动量
            if max_motion < 0.08 or avg_motion < 0.04:
                print(f"警告：{self.current_gesture}动作幅度过小 (最大: {max_motion:.4f}, 平均: {avg_motion:.4f})")
                return False
        else:  # click, pinch
            # 点击和捏合需要较小的运动量
            if max_motion < 0.06 or avg_motion < 0.03:
                print(f"警告：{self.current_gesture}动作幅度过小 (最大: {max_motion:.4f}, 平均: {avg_motion:.4f})")
                return False

        # 检查动作完整性 - 计算整个过程中的最大相对位移
        def get_relative_fingertip_distance(frame_data):
            # 获取掌根坐标(0号点)
            palm = np.array(frame_data[0:2])  # 只取x,y坐标
            
            # 获取指尖坐标并计算相对位移
            index_finger = np.array(frame_data[8*3:8*3+2]) - palm  # 食指
            middle_finger = np.array(frame_data[12*3:12*3+2]) - palm  # 中指
            ring_finger = np.array(frame_data[16*3:16*3+2]) - palm  # 无名指
            
            # 计算加权平均
            return 0.5 * index_finger + 0.25 * middle_finger + 0.25 * ring_finger

        # 计算初始位置
        initial_position = get_relative_fingertip_distance(self.frames_buffer[0][0])
        
        # 计算整个过程中的最大位移
        max_distance = 0
        for frame_data in self.frames_buffer:
            current_position = get_relative_fingertip_distance(frame_data[0])
            distance = np.linalg.norm(current_position - initial_position)
            max_distance = max(max_distance, distance)
        
        # 根据手势类型设置不同的阈值
        if self.current_gesture in ["right_swipe", "left_swipe"]:
            if max_distance < 0.2:  # 滑动类手势需要较大的相对位移
                print(f"警告：{self.current_gesture}最大相对位移过小 (距离: {max_distance:.4f})")
                return False
        elif self.current_gesture in ["up_swipe", "down_swipe"]:
            if max_distance < 0.15:  # 上下滑动需要适中的相对位移
                print(f"警告：{self.current_gesture}最大相对位移过小 (距离: {max_distance:.4f})")
                return False
        else:  # click, pinch
            if max_distance < 0.05:  # 点击和捏合需要较小的相对位移
                print(f"警告：{self.current_gesture}最大相对位移过小 (距离: {max_distance:.4f})")
                return False

        print(f"数据质量验证通过: {self.current_gesture}")
        print(f"最大运动量: {max_motion:.4f}")
        print(f"平均运动量: {avg_motion:.4f}")
        print(f"最大相对位移: {max_distance:.4f}")
        return True

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