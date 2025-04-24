import sys
import cv2
import numpy as np
import os
import tensorflow as tf
from collections import deque
import mediapipe as mp
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel,
                            QPushButton, QLineEdit, QVBoxLayout,
                            QWidget, QHBoxLayout, QMessageBox, QCheckBox, QRadioButton,
                            QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QResizeEvent
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread, QMutex, QSize
# 导入自定义模块
from recognition import GestureRecognition
from mouseControl import MouseControlThread
from handsControl import HandsControlThread  # 导入新的手势控制类

# 与train.py保持一致的配置
# 提前定义手势名称列表，确保顺序一致
GESTURE_NAMES = ["right_swipe", "left_swipe", "up_swipe", "down_swipe", "click", "pinch", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
# 选定的关键点ID - 掌根(0)、拇指(4)、食指(5,8)、中指(9,12)、无名指(13,16)、小指(17,20)
SELECTED_LANDMARKS = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]
# 更新输入维度
INPUT_DIM = len(SELECTED_LANDMARKS) * 3  # 10个关键点 × 3坐标 = 30
# 序列长度设置
SEQUENCE_LENGTH = 30  # 与train.py保持一致

# 尝试检测和配置GPU
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"检测到 {len(gpus)} 个GPU设备")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("已启用GPU内存动态增长")
    else:
        print("未检测到GPU设备，将使用CPU")
except Exception as e:
    print(f"GPU设置错误: {str(e)}")

class CameraThread(QThread):
    """
    摄像头线程类：负责处理摄像头捕获和基础处理
    """
    raw_frame_signal = pyqtSignal(np.ndarray)  # 原始帧信号
    processed_frame_signal = pyqtSignal(np.ndarray)  # 处理后帧信号
    error_signal = pyqtSignal(str)  # 错误信号
    camera_size_signal = pyqtSignal(int, int)  # 摄像头尺寸信号，用于通知UI
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.cap = None
        self.frame_mutex = QMutex()  # 帧互斥锁，防止同时访问
        self.latest_frame = None
        self.latest_processed_frame = None
        
        
        # MediaPipe设置 - 基本手部检测
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0  # 较低复杂度模型
        )
        
        # 性能优化
        self.frame_interval = 0.02  # 帧间隔控制(50fps)
        self.last_ui_update_time = 0
        self.ui_update_interval = 0.04  # UI更新间隔(25fps)
    
    def run(self):
        """线程主循环，捕获并处理摄像头帧"""
        try:
            # 打开摄像头
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.error_signal.emit(f"无法打开摄像头ID: {self.camera_id}")
                return
                
            # 设置摄像头参数 - 尝试设置高分辨率，但最终以实际支持为准
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 尝试设置更高分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # 获取实际的摄像头分辨率
            self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"实际摄像头分辨率: {self.camera_width}x{self.camera_height}")
            
            # 发送摄像头尺寸信号
            self.camera_size_signal.emit(self.camera_width, self.camera_height)
            
            # 主循环
            while self.running:
                # 捕获帧
                ret, frame = self.cap.read()
                if not ret:
                    self.error_signal.emit("无法读取摄像头画面")
                    break
                
                # 水平翻转（镜像效果）
                frame = cv2.flip(frame, 1)
                
                # 保存最新的原始帧
                self.frame_mutex.lock()
                self.latest_frame = frame.copy()
                self.frame_mutex.unlock()
                
                # 发送原始帧信号
                self.raw_frame_signal.emit(frame.copy())
                
                # 处理帧 - 基础手部检测
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # 如果检测到手部，绘制关键点
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                
                # 保存处理后的帧
                self.frame_mutex.lock()
                self.latest_processed_frame = frame.copy()
                self.frame_mutex.unlock()
                
                # 控制UI更新频率，减少频闪
                current_time = time.time()
                if current_time - self.last_ui_update_time >= self.ui_update_interval:
                    # 发送处理后的帧信号
                    self.processed_frame_signal.emit(frame.copy())
                    self.last_ui_update_time = current_time
                
                # 控制帧率
                time.sleep(self.frame_interval)
        
        except Exception as e:
            self.error_signal.emit(f"摄像头线程错误: {str(e)}")
        
        finally:
            # 释放资源
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
    
    def get_latest_frame(self):
        """获取最新的原始帧"""
        self.frame_mutex.lock()
        if self.latest_frame is not None:
            frame = self.latest_frame.copy()
        else:
            frame = None
        self.frame_mutex.unlock()
        return frame
    
    def get_latest_processed_frame(self):
        """获取最新的处理帧"""
        self.frame_mutex.lock()
        if self.latest_processed_frame is not None:
            frame = self.latest_processed_frame.copy()
        else:
            frame = None
        self.frame_mutex.unlock()
        return frame
    
    def start_camera(self, camera_id=None):
        """启动摄像头"""
        if camera_id is not None:
            self.camera_id = camera_id
        
        self.running = True
        if not self.isRunning():
            self.start()
    
    def stop_camera(self):
        """停止摄像头"""
        self.running = False
        
        # 等待线程结束
        if self.isRunning():
            self.wait()

class ActionState:
    NO_ACTION = 0
    ACTION_START = 1
    ACTION_IN_PROGRESS = 2
    ACTION_END = 3

class PredictionWorker(QObject):
    finished = pyqtSignal()
    result_ready = pyqtSignal(str, float)
    status_update = pyqtSignal(str)

    def __init__(self, model, sequence):
        super().__init__()
        self.model = model
        self.sequence = sequence
        self.is_running = False

    def predict(self):
        self.is_running = True
        try:
            self.status_update.emit("开始预测...")
            print("\n=== 开始预测 ===")
            print(f"输入数据形状: {self.sequence.shape}")
            
            # 确保输入数据格式正确
            if len(self.sequence.shape) == 2:
                input_data = np.expand_dims(self.sequence, axis=0)
            else:
                input_data = self.sequence
                
            print(f"模型输入形状: {input_data.shape}")
            self.status_update.emit("模型预测中...")
            
            # 进行预测
            pred = self.model.predict(input_data, verbose=0)
            print(f"预测结果形状: {pred.shape}")
            
            # 获取所有动作的置信度
            gesture_confidences = pred[0]
            max_idx = np.argmax(gesture_confidences)
            max_confidence = gesture_confidences[max_idx]
            max_gesture = GESTURE_NAMES[max_idx]  # 使用全局定义的GESTURE_NAMES
                     
            print(f"\n最大置信度动作: {max_gesture}")
            print(f"置信度: {max_confidence:.4f}")
            
            if self.is_running:
                confidence_threshold = 0.2  # 保持一致的置信度阈值
                if max_confidence > confidence_threshold:
                    self.status_update.emit(f"识别到: {max_gesture}")
                    self.result_ready.emit(max_gesture, float(max_confidence))
                    print(f"输出动作: {max_gesture} (置信度: {max_confidence:.4f})")
                else:
                    self.status_update.emit("置信度过低")
                    self.result_ready.emit("无动作", float(max_confidence))
                    print(f"输出: 无动作 (最大置信度 {max_confidence:.4f} 低于阈值 {confidence_threshold})")
            print("=== 预测结束 ===\n")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"预测出错: {str(e)}\n{error_details}")
            self.status_update.emit(f"预测错误: {str(e)}")
            if self.is_running:
                self.result_ready.emit("识别错误", 0.0)
        finally:
            self.is_running = False
            self.finished.emit()

    def stop(self):
        self.is_running = False

class HandGestureApp(QMainWindow):
    """
    手势识别应用主窗口
    """
    # 添加信号用于更新UI
    update_result_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(str)
    update_motion_signal = pyqtSignal(str, float)
    update_action_signal = pyqtSignal(str, float)
    send_frame_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        
        # 初始化状态变量
        self.camera_running = False
        self.recognition_running = False
        self.mouse_control_running = False
        self.display_mutex = QMutex()  # 用于线程安全的显示更新
        self.saved_sequence = None  # 保存的动作序列
        
        # 设置定时器
        self.frame_update_timer = QTimer(self)
        self.frame_update_timer.timeout.connect(self.update_frame_display)
        self.frame_update_timer.setInterval(20)  # 50fps，提高帧率，从25ms改为20ms
        
        # 应用状态
        self.last_update_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # 摄像头尺寸 - 初始化为默认值
        self.camera_width = 640
        self.camera_height = 480
        
        # 动作检测参数
        self.action_started = False  # 动作开始标志
        self.has_prediction = False  # 当前动作是否已经产生预测结果
        self.last_prediction = ""  # 上一次预测结果
        self.confirmation_count = 0  # 确认计数器
        self.current_gesture = "无动作"
        
        # 手势特定的阈值 - 优化检测灵敏度
        self.gesture_thresholds = {
            "right_swipe": {"motion": 0.08, "distance": 0.1, "stable_frames": 5},
            "left_swipe": {"motion": 0.08, "distance": 0.1, "stable_frames": 5},
            "up_swipe": {"motion": 0.08, "distance": 0.08, "stable_frames": 5},
            "down_swipe": {"motion": 0.08, "distance": 0.08, "stable_frames": 5},
            "click": {"motion": 0.06, "distance": 0.03, "stable_frames": 3},
            "pinch": {"motion": 0.06, "distance": 0.03, "stable_frames": 3},
            "one": {"motion": 0.06, "distance": 0.03, "stable_frames": 3},
            "two": {"motion": 0.06, "distance": 0.03, "stable_frames": 3},
            "three": {"motion": 0.06, "distance": 0.03, "stable_frames": 3},
            "four": {"motion": 0.06, "distance": 0.03, "stable_frames": 3},
            "five": {"motion": 0.06, "distance": 0.03, "stable_frames": 3},
            "six": {"motion": 0.06, "distance": 0.03, "stable_frames": 3},
            "seven": {"motion": 0.06, "distance": 0.03, "stable_frames": 3},
            "eight": {"motion": 0.06, "distance": 0.03, "stable_frames": 3},
            "nine": {"motion": 0.06, "distance": 0.03, "stable_frames": 3},
            "ten": {"motion": 0.06, "distance": 0.03, "stable_frames": 3}
        }
        
        # 动作检测参数
        self.initial_position = None  # 初始位置
        self.max_distance = 0  # 最大相对位移
        self.motion_buffer = deque(maxlen=100)  # 运动量缓冲区
        self.position_buffer = deque(maxlen=100)  # 位置缓冲区
        self.stable_frames = 0  # 静止帧计数器
        
        # 使用默认阈值初始化
        self.current_thresholds = self.gesture_thresholds["right_swipe"]
        
        # 序列数据缓冲区
        self.sequence_length = SEQUENCE_LENGTH
        self.data_buffer = deque(maxlen=self.sequence_length)
        
        # 创建线程
        self.camera_thread = CameraThread(camera_id=0)
        self.gesture_recognition = GestureRecognition()
        self.mouse_control = MouseControlThread()
        self.hands_control = HandsControlThread()  # 添加手势控制线程
        
        # 设置UI
        self.init_ui()
        
        # 连接信号
        self.connect_signals()
        
        # 添加连接从main1.py导入的信号
        self.update_status_signal.connect(self.update_status)
        self.update_motion_signal.connect(self.update_motion_display)
        self.update_action_signal.connect(self.update_action_display)
        self.update_result_signal.connect(self.update_result_display)
        self.send_frame_signal.connect(self.handle_shared_frame)
        
        print("应用初始化完成")
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("手势控制应用")
        self.setGeometry(100, 100, 800, 600)
        
        # 创建主布局
        main_layout = QVBoxLayout()
        
        # 创建显示区域 - 修改为自适应窗口大小
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        
        # 设置初始尺寸为摄像头尺寸，允许缩放
        self.display_label.setMinimumSize(self.camera_width // 2, self.camera_height // 2)
        self.display_label.setMaximumSize(self.camera_width * 2, self.camera_height * 2)
        self.display_label.setSizePolicy(
            QSizePolicy.Expanding, 
            QSizePolicy.Expanding
        )
        self.display_label.setStyleSheet("border: 2px solid gray;")
        main_layout.addWidget(self.display_label, 1)  # 添加拉伸因子
        
        # 创建状态标签
        self.status_label = QLabel("就绪")
        main_layout.addWidget(self.status_label)
        
        # 创建结果标签 - 从main1.py引入
        self.result_label = QLabel("等待识别...", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 24px; color: red;")
        main_layout.addWidget(self.result_label)
        
        # 添加加载状态标签 - 从main1.py引入
        self.loading_label = QLabel("", self)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("font-size: 18px; color: blue; font-weight: bold;")
        main_layout.addWidget(self.loading_label)
        
        # 添加置信度标签 - 从main1.py引入
        self.confidence_label = QLabel("置信度: -", self)
        self.confidence_label.setAlignment(Qt.AlignLeft)
        self.confidence_label.setStyleSheet("font-size: 16px; color: black;")
        main_layout.addWidget(self.confidence_label)
        
        # 创建控制按钮组
        button_layout = QHBoxLayout()
        
        # 摄像头ID输入
        self.camera_id_input = QLineEdit("0")
        self.camera_id_input.setPlaceholderText("摄像头ID")
        self.camera_id_input.setFixedWidth(50)
        button_layout.addWidget(self.camera_id_input)
        
        # 摄像头控制按钮
        self.camera_button = QPushButton("启动摄像头")
        self.camera_button.clicked.connect(self.toggle_camera)
        button_layout.addWidget(self.camera_button)
        
        # 手势识别控制按钮
        self.recognition_button = QPushButton("启动手势识别")
        self.recognition_button.clicked.connect(self.toggle_recognition)
        self.recognition_button.setEnabled(False)  # 初始禁用
        button_layout.addWidget(self.recognition_button)
        
        # 鼠标控制按钮
        self.mouse_control_button = QPushButton("启动鼠标控制")
        self.mouse_control_button.clicked.connect(self.toggle_mouse_control)
        self.mouse_control_button.setEnabled(False)  # 初始禁用
        button_layout.addWidget(self.mouse_control_button)
        
        main_layout.addLayout(button_layout)
        
        # 创建中央部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # 允许窗口大小调整时更新布局
        self.setMinimumSize(640, 480)
    
    def connect_signals(self):
        """连接信号和槽"""
        # 连接摄像头线程信号 - 改为间接连接，避免直接更新UI
        self.camera_thread.raw_frame_signal.connect(self.process_raw_frame)
        self.camera_thread.error_signal.connect(self.show_error)
        self.camera_thread.camera_size_signal.connect(self.update_camera_size)
        
        # 连接手势识别信号
        self.gesture_recognition.status_signal.connect(self.update_status)
        self.gesture_recognition.gesture_signal.connect(self.process_gesture_result)
        
        # 连接鼠标控制信号
        self.mouse_control.status_signal.connect(self.update_status)
        self.mouse_control.frame_signal.connect(self.update_mouse_frame)
        
        # 连接手势控制信号
        self.hands_control.status_signal.connect(self.update_status)
        self.hands_control.mode_switch_signal.connect(self.switch_control_mode)
    
    def toggle_camera(self):
        """切换摄像头状态"""
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """启动摄像头"""
        try:
            camera_id = int(self.camera_id_input.text())
            self.camera_thread.start_camera(camera_id)
            self.camera_running = True
            self.camera_button.setText("停止摄像头")
            self.recognition_button.setEnabled(True)
            self.mouse_control_button.setEnabled(True)
            self.update_status("摄像头已启动")
            
            # 启动帧更新定时器
            self.frame_update_timer.start()
        except Exception as e:
            self.show_error(f"启动摄像头错误: {str(e)}")
    
    def stop_camera(self):
        """停止摄像头"""
        try:
            # 先停止依赖摄像头的功能
            if self.recognition_running:
                self.stop_recognition()
            
            if self.mouse_control_running:
                self.stop_mouse_control()
            
            # 停止帧更新定时器
            self.frame_update_timer.stop()
            
            # 停止摄像头线程
            self.camera_thread.stop_camera()
            self.camera_running = False
            
            # 更新UI
            self.camera_button.setText("启动摄像头")
            self.recognition_button.setEnabled(False)
            self.mouse_control_button.setEnabled(False)
            self.display_label.clear()
            self.update_status("摄像头已停止")
        except Exception as e:
            self.show_error(f"停止摄像头错误: {str(e)}")

    def toggle_recognition(self):
        """切换手势识别状态"""
        if not self.recognition_running:
            self.start_recognition()
        else:
            self.stop_recognition()

    def start_recognition(self):
        """启动手势识别"""
        try:
            # 先停止鼠标控制（互斥操作）
            if self.mouse_control_running:
                self.stop_mouse_control()
            
            # 启动手势识别
            self.gesture_recognition.start_recognition()
            self.recognition_running = True
            self.recognition_button.setText("停止手势识别")
            self.update_status("手势识别已启动")
            
            # 同时启动手势控制
            self.hands_control.start_control()
            self.update_status("手势控制已启动")
        except Exception as e:
            self.show_error(f"启动手势识别错误: {str(e)}")

    def stop_recognition(self):
        """停止手势识别"""
        try:
            self.gesture_recognition.stop_recognition()
            self.recognition_running = False
            self.recognition_button.setText("启动手势识别")
            self.update_status("手势识别已停止")
            
            # 同时停止手势控制
            self.hands_control.stop_control()
            self.update_status("手势控制已停止")
        except Exception as e:
            self.show_error(f"停止手势识别错误: {str(e)}")
    
    def toggle_mouse_control(self):
        """切换鼠标控制状态"""
        if not self.mouse_control_running:
            self.start_mouse_control()
        else:
            self.stop_mouse_control()
    
    def start_mouse_control(self):
        """启动鼠标控制"""
        try:
            # 先停止手势识别（互斥操作）
            if self.recognition_running:
                self.stop_recognition()
            
            # 启动鼠标控制
            self.mouse_control.start_control()
            self.mouse_control_running = True
            self.mouse_control_button.setText("停止鼠标控制")
            self.update_status("鼠标控制已启动")
        except Exception as e:
            self.show_error(f"启动鼠标控制错误: {str(e)}")
    
    def stop_mouse_control(self):
        """停止鼠标控制"""
        try:
            self.mouse_control.stop_control()
            self.mouse_control_running = False
            self.mouse_control_button.setText("启动鼠标控制")
            self.update_status("鼠标控制已停止")
        except Exception as e:
            self.show_error(f"停止鼠标控制错误: {str(e)}")
    
    def process_raw_frame(self, frame):
        """处理原始帧 - 转发给需要的组件，添加动作检测逻辑"""
        try:
            # 计算FPS
            current_time = time.time()
            self.frame_count += 1
            
            # 每秒更新一次FPS
            if current_time - self.last_update_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_update_time = current_time
            
            # 为鼠标控制共享帧 - 减少不必要的帧复制操作
            if self.mouse_control_running:
                try:
                    # 每2帧处理一次以减少计算负担，同时保持足够的响应性
                    if self.frame_count % 2 == 0:
                        # 注意：此处不再创建帧的副本，而是直接传递原始帧
                        # 由接收方负责创建副本（如果需要）
                        self.send_frame_signal.emit(frame)
                except Exception as e:
                    print(f"帧传递错误: {str(e)}")
            
            # 手势识别处理
            if self.recognition_running:
                # 创建帧副本供手势识别使用
                frame_copy = frame.copy()
                
                # 手部检测和手势识别处理
                # MediaPipe手部检测
                image_height, image_width, _ = frame_copy.shape
                rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                results = self.gesture_recognition.hands.process(rgb_frame)
                
                # 如果检测到手部，进行处理
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 绘制手部关键点
                        self.gesture_recognition.mp_drawing.draw_landmarks(
                            frame_copy,
                            hand_landmarks,
                            self.gesture_recognition.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # 从main1.py导入的手势处理逻辑
                        # 处理关键点数据
                        frame_data = self.process_landmarks(
                            hand_landmarks,
                            image_width,
                            image_height
                        )
                        
                        # 计算当前帧的运动量
                        current_motion = self.calculate_motion(frame_data)
                        self.motion_buffer.append(current_motion)
                        self.position_buffer.append(frame_data)
                        
                        # 更新UI显示当前运动量 - 使用直接方式更新UI
                        try:
                            # 每3帧更新一次UI，减少UI更新频率
                            if self.frame_count % 3 == 0:
                                self.update_motion_signal.emit("运动量", current_motion)
                        except Exception as e:
                            print(f"UI更新错误: {str(e)}")
                        
                        # 动作检测逻辑
                        if not self.action_started:
                            if current_motion > self.current_thresholds["motion"]:
                                self.action_started = True
                                self.has_prediction = False  # 重置预测状态
                                self.initial_position = self.calculate_relative_distance(frame_data)
                                self.max_distance = 0
                                self.data_buffer.clear()  # 清空之前的数据
                                print(f"\n检测到动作开始 (运动量: {current_motion:.4f})")
                                self.update_action_signal.emit("检测到动作开始", current_motion)
                        
                        if self.action_started:
                            # 更新最大相对位移
                            current_relative = self.calculate_relative_distance(frame_data)
                            current_distance = np.linalg.norm(current_relative - self.initial_position)
                            self.max_distance = max(self.max_distance, current_distance)
                            
                            # 更新UI显示当前动作状态
                            self.update_action_signal.emit("动作进行中", self.max_distance)
                            
                            self.data_buffer.append(frame_data)
                            
                            # 检测动作结束
                            if current_motion < self.current_thresholds["motion"]/2:
                                self.stable_frames += 1
                                if self.stable_frames >= self.current_thresholds["stable_frames"]:
                                    # 检查动作完整性
                                    if self.max_distance >= self.current_thresholds["distance"]:
                                        self.action_started = False
                                        self.stable_frames = 0
                                        print(f"检测到动作结束 (最大相对位移: {self.max_distance:.4f})")
                                        self.update_action_signal.emit("检测到动作结束", self.max_distance)
                                        if len(self.data_buffer) >= 15:  # 至少收集15帧才进行预测
                                            # 标记动作已经开始预测
                                            self.has_prediction = True
                                            self.run_prediction()
                                        else:
                                            # 动作帧数不足，无法预测，标记为无动作
                                            print("帧数不足，无法预测")
                                            self.update_result_signal.emit("无动作")
                                            self.update_motion_signal.emit("帧数不足", 0.0)
                                    else:
                                        print(f"动作不完整，放弃预测 (最大相对位移: {self.max_distance:.4f})")
                                        self.update_action_signal.emit("动作不完整，放弃预测", self.max_distance)
                                        # 标记为无动作
                                        self.update_result_signal.emit("无动作")
                                        self.update_motion_signal.emit("位移不足", self.max_distance)
                                        self.action_started = False
                                        self.stable_frames = 0
                                        self.data_buffer.clear()
                            else:
                                self.stable_frames = 0
                else:
                    # 如果没有检测到手，重置状态
                    if self.action_started:
                        self.action_started = False
                        self.data_buffer.clear()
                    
                    # 降低无手势状态下的UI更新频率
                    if self.frame_count % 5 == 0:
                        self.update_motion_signal.emit("无手势", 0.0)
                        self.update_action_signal.emit("等待手势", 0.0)
                
                # 将处理后的帧传递给手势识别模块 - 使用直接方式传递帧
                try:
                    # 由于手势识别处理相对耗时，减少处理频率
                    if self.frame_count % 2 == 0:
                        self.gesture_recognition.process_shared_frame(frame_copy)
                except Exception as e:
                    print(f"帧处理错误: {str(e)}")
            
        except Exception as e:
            self.show_error(f"处理帧错误: {str(e)}")
    
    def process_landmarks(self, hand_landmarks, frame_width, frame_height):
        """处理手部关键点数据，添加标准化
        
        Args:
            hand_landmarks: MediaPipe检测到的手部关键点
            frame_width: 帧宽度
            frame_height: 帧高度
            
        Returns:
            处理后的关键点数据
        """
        # 转换为相对坐标
        root = hand_landmarks.landmark[0]
        frame_data = []
        for lm in hand_landmarks.landmark:
            frame_data.extend([
                lm.x - root.x,  # 相对X坐标
                lm.y - root.y,  # 相对Y坐标
                lm.z - root.z  # 相对Z坐标
            ])
        
        # 数据标准化
        frame_data = np.array(frame_data)
        frame_data = (frame_data - np.mean(frame_data)) / (np.std(frame_data) + 1e-8)
        
        # 提取选定的关键点
        selected_frame_data = self.extract_landmarks(frame_data)
        
        return selected_frame_data
    
    def extract_landmarks(self, frame_data):
        """从完整的帧数据中提取选定的关键点
        
        Args:
            frame_data: 形状为 (21*3) 的原始数据，每3个值代表一个关键点的x,y,z坐标
            
        Returns:
            形状为 (len(SELECTED_LANDMARKS)*3) 的处理后数据
        """
        # 将数据重新整形为21x3
        reshaped_data = np.array(frame_data).reshape(21, 3)
        
        # 提取选定的关键点
        selected_data = reshaped_data[SELECTED_LANDMARKS, :]
        
        # 展平回一维数组
        return selected_data.flatten().tolist()
    
    def calculate_motion(self, current_data):
        """计算最近两帧之间的运动量"""
        if len(self.position_buffer) < 2:
            return 0
        prev_data = np.array(self.position_buffer[-2])
        curr_data = np.array(current_data)
        motion = np.mean(np.abs(curr_data - prev_data))
        return motion
    
    def calculate_relative_distance(self, frame_data):
        """计算指尖相对于掌根的位移
        
        由于我们使用了选定的关键点，需要调整索引
        SELECTED_LANDMARKS = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]
        对应的是：掌根(0)、拇指尖(4)、食指根(5)、中指根(9)、无名指根(13)、小指根(17)、
                食指尖(8)、中指尖(12)、无名指尖(16)、小指尖(20)
        在选定关键点数组中的索引为：
        掌根=0, 食指尖=6, 中指尖=7, 无名指尖=8, 小指尖=9
        """
        # 获取掌根坐标(0号点)
        palm = np.array(frame_data[0:2])  # 只取x,y坐标
        
        # 获取当前手势类型，以决定如何计算相对位移
        if hasattr(self, 'current_gesture') and self.current_gesture in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]:
            # 对于数字手势，关注所有指尖的位置
            thumb_tip = np.array(frame_data[1*3:1*3+2]) - palm  # 拇指尖（索引1对应原始关键点4）
            index_finger = np.array(frame_data[6*3:6*3+2]) - palm  # 食指尖
            middle_finger = np.array(frame_data[7*3:7*3+2]) - palm  # 中指尖
            ring_finger = np.array(frame_data[8*3:8*3+2]) - palm  # 无名指尖
            pinky_finger = np.array(frame_data[9*3:9*3+2]) - palm  # 小指尖
            
            # 综合考虑所有手指的位置
            return 0.2 * thumb_tip + 0.2 * index_finger + 0.2 * middle_finger + 0.2 * ring_finger + 0.2 * pinky_finger
        else:
            # 获取指尖坐标并计算相对位移
            # 索引6是食指尖(对应原始关键点8)
            index_finger = np.array(frame_data[6*3:6*3+2]) - palm  # 食指尖
            # 索引7是中指尖(对应原始关键点12)
            middle_finger = np.array(frame_data[7*3:7*3+2]) - palm  # 中指尖
            # 索引8是无名指尖(对应原始关键点16)
            ring_finger = np.array(frame_data[8*3:8*3+2]) - palm  # 无名指尖
            
            # 计算加权平均
            return 0.5 * index_finger + 0.25 * middle_finger + 0.25 * ring_finger
    
    def run_prediction(self):
        """运行预测逻辑"""
        try:
            # 显示加载状态
            self.update_status_signal.emit("正在准备数据...")
            
            # 动态填充/截断序列
            sequence = np.array(list(self.data_buffer))
            if len(sequence) < self.sequence_length:
                # 重复最后一帧填充不足部分
                last_frame = sequence[-1]
                pad_frames = np.tile(last_frame, (self.sequence_length-len(sequence), 1))
                sequence = np.concatenate([sequence, pad_frames])
            else:
                sequence = sequence[-self.sequence_length:]  # 取最后30帧
            
            # 准备预测
            self.update_status_signal.emit("开始预测...")
            print(f"\n开始预测 (收集帧数: {len(self.data_buffer)})")
            
            # 将序列传递给手势识别模块进行预测
            self.gesture_recognition.predict_gesture(sequence)
        except Exception as e:
            print(f"创建预测时出错: {str(e)}")
            self.update_status_signal.emit(f"预测错误: {str(e)}")
    
    def process_gesture_result(self, gesture, confidence):
        """处理手势识别结果"""
        try:
            # 确保UI更新在主线程中执行
            QApplication.processEvents()
            
            # 更新置信度显示
            self.confidence_label.setText(f"置信度: {confidence:.4f}")
            
            # 更新当前手势的阈值
            if gesture in self.gesture_thresholds:
                self.current_thresholds = self.gesture_thresholds[gesture]
                print(f"更新阈值: {gesture}")
                print(f"运动量阈值: {self.current_thresholds['motion']}")
                print(f"位移阈值: {self.current_thresholds['distance']}")
                print(f"静止帧阈值: {self.current_thresholds['stable_frames']}")
            
            # 直接更新结果显示
            if confidence > 0.2:  # 置信度阈值
                # 标记已经有了有效的预测结果
                self.has_prediction = True
                self.update_result_signal.emit(gesture)
                self.current_gesture = gesture
                self.last_prediction = gesture
                self.confirmation_count = 1
                # 立即更新可能的手势
                self.update_motion_signal.emit(f"检测到: {gesture}", confidence)
                
                # 将手势传递给手势控制模块处理
                self.hands_control.process_gesture(gesture, confidence)
            else:
                self.update_result_signal.emit("无动作")
                self.update_motion_signal.emit("置信度过低", confidence)
            
            # 如果鼠标控制正在运行，将手势结果传递给它
            if self.mouse_control_running and hasattr(self.mouse_control, 'process_gesture'):
                self.mouse_control.process_gesture(gesture, confidence)
                
            # 确保UI立即更新
            QApplication.processEvents()
        except Exception as e:
            self.show_error(f"处理手势结果错误: {str(e)}")
    
    def update_motion_display(self, text, value):
        """更新动作状态显示"""
        try:
            self.status_label.setText(f"状态: {text} ({value:.4f})")
            # 确保UI立即更新，但不阻塞处理
            QApplication.processEvents()
        except Exception as e:
            print(f"更新动作状态显示出错: {str(e)}")
        
    def update_action_display(self, text, value):
        """更新动作进度显示"""
        try:
            self.loading_label.setText(f"{text} - 最大位移: {value:.4f}")
            # 确保UI立即更新，但不阻塞处理
            QApplication.processEvents()
        except Exception as e:
            print(f"更新动作进度显示出错: {str(e)}")
    
    def update_result_display(self, text):
        """更新结果显示"""
        try:
            self.result_label.setText(text)
            # 根据结果改变颜色
            color = "green" if text != "无动作" else "red"
            self.result_label.setStyleSheet(f"font-size: 24px; color: {color};")
            
            # 确保UI立即更新，但不阻塞处理
            QApplication.processEvents()
        except Exception as e:
            print(f"更新结果显示出错: {str(e)}")
    
    def update_mouse_frame(self, frame):
        """更新鼠标控制模式的帧显示"""
        # 直接更新鼠标控制帧，不再检查display_priority
        try:
            h, w, c = frame.shape
            # 先转换为QImage，再创建QPixmap
            q_img = QImage(frame.data, w, h, w * c, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            
            # 获取显示标签的当前大小
            label_size = self.display_label.size()
            
            # 计算保持宽高比的缩放后尺寸
            scaled_pixmap = pixmap.scaled(
                label_size, 
                Qt.KeepAspectRatio,  # 保持原始宽高比
                Qt.SmoothTransformation  # 使用平滑变换提高质量
            )
            
            # 更新显示标签
            self.display_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"更新鼠标控制帧错误: {str(e)}")

    def handle_shared_frame(self, frame):
        """处理从手势识别共享的帧"""
        # 确保鼠标控制线程正在运行且处于共享模式
        if self.mouse_control_running and hasattr(self.mouse_control, 'process_shared_frame'):
            try:
                # 直接传递帧给鼠标控制线程处理，不进行额外复制
                self.mouse_control.process_shared_frame(frame)
            except Exception as e:
                print(f"共享帧错误: {str(e)}")
    
    def update_frame_display(self):
        """定时更新帧显示 - 避免多线程频繁更新导致的UI闪烁"""
        try:
            frame = None
            
            # 优先显示当前正在运行的功能的帧
            if self.recognition_running:
                # 获取手势识别处理后的帧
                frame = self.gesture_recognition.get_latest_frame()
            elif self.mouse_control_running:
                # 获取鼠标控制处理后的帧
                frame = self.mouse_control.get_latest_frame()
            else:
                # 获取摄像头处理后的帧
                frame = self.camera_thread.get_latest_processed_frame()
            
            # 如果没有可用帧，退出
            if frame is None:
                return
                
            # 使用PIL处理中文显示
            try:
                from PIL import Image, ImageDraw, ImageFont
                import numpy as np
                
                # 转换为PIL图像以支持中文
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                
                # 尝试加载中文字体
                try:
                    # 增加更多可能的字体路径选择
                    font_candidates = [
                        os.path.join(os.getenv('SystemRoot', 'C:\\Windows'), 'Fonts', 'simhei.ttf'),
                        os.path.join(os.getenv('SystemRoot', 'C:\\Windows'), 'Fonts', 'msyh.ttf'),
                        os.path.join(os.getenv('SystemRoot', 'C:\\Windows'), 'Fonts', 'simsun.ttc'),
                        os.path.join(os.getenv('SystemRoot', 'C:\\Windows'), 'Fonts', 'arialuni.ttf'),
                        'simhei.ttf',
                        'msyh.ttf',
                        'simsun.ttc'
                    ]
                    
                    font_found = False
                    for font_path in font_candidates:
                        if os.path.exists(font_path):
                            try:
                                fps_font = ImageFont.truetype(font_path, 24)
                                mode_font = ImageFont.truetype(font_path, 20)
                                font_found = True
                                break
                            except Exception:
                                continue
                            
                    if not font_found:
                        fps_font = ImageFont.load_default()
                        mode_font = ImageFont.load_default()
                except Exception:
                    fps_font = ImageFont.load_default()
                    mode_font = ImageFont.load_default()
                
                # 添加FPS显示
                draw.text((10, 30), f"FPS: {self.fps}", font=fps_font, fill=(0, 255, 0))
                
                # 添加模式显示
                status_text = ""
                if self.recognition_running:
                    status_text = "Mode: Gesture Recognition"
                elif self.mouse_control_running:
                    status_text = "Mode: Mouse Control"
                else:
                    status_text = "Mode: Camera Only"
                
                draw.text((10, 60), status_text, font=mode_font, fill=(255, 255, 0))
                
                # 转换回OpenCV格式
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
            except ImportError:
                # 如果PIL不可用，回退到OpenCV文本渲染
                cv2.putText(
                    frame, 
                    f"FPS: {self.fps}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                # 添加模式显示
                status_text = ""
                if self.recognition_running:
                    status_text = "Mode: Gesture Recognition"
                elif self.mouse_control_running:
                    status_text = "Mode: Mouse Control"
                else:
                    status_text = "Mode: Camera Only"
                
                cv2.putText(
                    frame, 
                    status_text, 
                    (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 0), 
                    2
                )
            
            # 转换帧到QPixmap并显示
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            
            # 使用锁确保QImage创建和显示过程不被中断
            try:
                # 创建QImage
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_img)
                
                # 获取显示标签的当前大小
                label_size = self.display_label.size()
                
                # 计算保持宽高比的缩放后尺寸
                scaled_pixmap = pixmap.scaled(
                    label_size, 
                    Qt.KeepAspectRatio,  # 保持原始宽高比
                    Qt.SmoothTransformation  # 使用平滑变换提高质量
                )
                
                # 更新显示标签
                self.display_label.setPixmap(scaled_pixmap)
                
                # 居中显示图像
                self.display_label.setAlignment(Qt.AlignCenter)
            except Exception as e:
                print(f"更新图像显示出错: {str(e)}")
                
        except Exception as e:
            if "get_latest_frame" in str(e):
                # 忽略特定错误，等待下一次更新
                pass
            else:
                self.show_error(f"更新显示错误: {str(e)}")
    
    def update_status(self, message):
        """更新状态栏消息"""
        self.status_label.setText(message)
    
    def show_error(self, error_message):
        """显示错误消息"""
        print(f"错误: {error_message}")
        self.update_status(f"错误: {error_message}")
    
    def closeEvent(self, event):
        """关闭窗口事件处理"""
        try:
            print("正在关闭应用...")
            # 停止各个线程
            if self.camera_thread is not None:
                self.stop_camera()
                try:
                    # 等待最多0.5秒以确保线程停止
                    self.camera_thread.wait(500)
                except Exception as e:
                    print(f"等待摄像头线程停止时出错: {str(e)}")
            
            if self.gesture_recognition is not None:
                self.stop_recognition()
                try:
                    # 等待最多0.5秒以确保线程停止
                    self.gesture_recognition.wait(500)
                except Exception as e:
                    print(f"等待手势识别线程停止时出错: {str(e)}")
            
            if self.mouse_control is not None:
                self.stop_mouse_control()
                try:
                    # 等待最多0.5秒以确保线程停止
                    self.mouse_control.wait(500)
                except Exception as e:
                    print(f"等待鼠标控制线程停止时出错: {str(e)}")
            
            if self.hands_control is not None:
                self.hands_control.stop_control()
                try:
                    # 等待最多0.5秒以确保线程停止
                    self.hands_control.wait(500)
                except Exception as e:
                    print(f"等待手势控制线程停止时出错: {str(e)}")
                    
            # 等待一小段时间确保所有资源都被清理
            time.sleep(0.2)
            
            # 保存设置
            self.save_settings()
            
            print("应用已安全关闭")
            event.accept()
        except Exception as e:
            print(f"关闭应用时出错: {str(e)}")
            event.accept()

    def resizeEvent(self, event: QResizeEvent):
        """处理窗口大小变化事件"""
        # 调用父类的resize事件处理
        super().resizeEvent(event)
        
        # 立即更新预览图像以适应新的窗口大小
        self.update_frame_display()
        
        # 记录窗口大小变化
        old_size = event.oldSize()
        new_size = event.size()
        if old_size.width() > 0 and old_size.height() > 0:  # 避免初始化时的无效事件
            print(f"窗口大小从 {old_size.width()}x{old_size.height()} 变为 {new_size.width()}x{new_size.height()}")

    def save_settings(self):
        """保存应用设置"""
        try:
            print("保存应用设置...")
            # 这里可以添加设置保存代码，如果需要
            # 例如保存摄像头ID等设置
        except Exception as e:
            print(f"保存设置出错: {str(e)}")

    def update_camera_size(self, width, height):
        """更新摄像头尺寸和UI显示区域"""
        self.camera_width = width
        self.camera_height = height
        
        # 根据摄像头实际尺寸调整显示区域大小
        self.display_label.setMinimumSize(width // 2, height // 2)
        self.display_label.setMaximumSize(width * 2, height * 2)
        
        # 更新状态
        self.update_status(f"摄像头分辨率: {width}x{height}")

    def switch_control_mode(self, mouse_mode):
        """切换控制模式
        
        Args:
            mouse_mode: True表示切换到鼠标模式，False表示切换到手势控制模式
        """
        try:
            if mouse_mode:
                # 切换到鼠标模式
                self.update_status("切换到鼠标控制模式")
                print("主界面响应：切换到鼠标控制模式")
                
                # 优化：直接调用鼠标控制，避免额外的UI更新和事件处理
                if not self.mouse_control_running:
                    # 直接启动鼠标控制线程
                    self.mouse_control.start_control()
                    self.mouse_control_running = True
                    self.mouse_control_button.setText("停止鼠标控制")
                    
                    # 添加调试信息
                    print("鼠标控制线程已启动")
            else:
                # 切换到手势控制模式
                self.update_status("切换到手势控制模式")
                print("主界面响应：切换到手势控制模式")
                
                # 优化：直接调用停止鼠标控制
                if self.mouse_control_running:
                    # 直接停止鼠标控制线程
                    self.mouse_control.stop_control()
                    self.mouse_control_running = False
                    self.mouse_control_button.setText("启动鼠标控制")
                    
                    # 添加调试信息
                    print("鼠标控制线程已停止")
            
            # 优化：批量处理UI更新，减少重绘次数
            QApplication.processEvents()
            
        except Exception as e:
            self.show_error(f"切换控制模式错误: {str(e)}")
            print(f"切换控制模式错误: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandGestureApp()
    window.show()
    sys.exit(app.exec_())