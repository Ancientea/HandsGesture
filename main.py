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

# 与train.py保持一致的配置
# 提前定义手势名称列表，确保顺序一致
GESTURE_NAMES = ["right_swipe", "left_swipe", "up_swipe", "down_swipe", "click", "pinch"]
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
            
            # 输出所有动作的置信度
            self.status_update.emit("处理预测结果...")
            print("\n各动作置信度:")
            for i, (gesture, conf) in enumerate(zip(GESTURE_NAMES, gesture_confidences)):
                print(f"{gesture}: {conf:.4f}")
            
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
    update_result_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(str)
    update_motion_signal = pyqtSignal(str, float)
    update_action_signal = pyqtSignal(str, float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("手势识别系统 (选定关键点)")
        self.setGeometry(100, 100, 800, 600)
        self.cap = None
        self.is_recognizing = False
        self.worker = None
        self.worker_thread = None
        # 使用预定义的GESTURE_NAMES
        self.gesture_names = GESTURE_NAMES
        self.init_ui()
        self.init_mediapipe()
        self.init_model()
        self.init_data_buffer()
        self.init_action_detection()
        
        # 添加信号连接
        self.update_status_signal.connect(self.update_status_display)
        self.update_motion_signal.connect(self.update_motion_display)
        self.update_action_signal.connect(self.update_action_display)
        
        # 尝试加载配置
        self.load_landmark_config()

    def load_landmark_config(self):
        """尝试加载关键点配置"""
        try:
            if os.path.exists('landmark_config.txt'):
                with open('landmark_config.txt', 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('SELECTED_LANDMARKS'):
                            # 这只是显示信息，实际SELECTED_LANDMARKS已在文件顶部定义
                            print(f"加载关键点配置: {line.strip()}")
                print("已加载关键点配置")
        except Exception as e:
            print(f"加载关键点配置时出错: {str(e)}")

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
        
        # 数据标准化 - 与train.py中的preprocess_sequence保持一致
        frame_data = np.array(frame_data)
        frame_data = (frame_data - np.mean(frame_data)) / (np.std(frame_data) + 1e-8)
        
        # 提取选定的关键点
        selected_frame_data = self.extract_landmarks(frame_data)
        
        return selected_frame_data

    def init_action_detection(self):
        """初始化动作检测相关参数"""
        self.action_started = False  # 动作开始标志
        self.last_prediction = ""  # 上一次预测结果
        self.confirmation_count = 0  # 确认计数器
        
        # 手势特定的阈值
        self.gesture_thresholds = {
            "right_swipe": {"motion": 0.08, "distance": 0.2, "stable_frames": 5},
            "left_swipe": {"motion": 0.08, "distance": 0.2, "stable_frames": 5},
            "up_swipe": {"motion": 0.08, "distance": 0.15, "stable_frames": 5},
            "down_swipe": {"motion": 0.08, "distance": 0.15, "stable_frames": 5},
            "click": {"motion": 0.06, "distance": 0.05, "stable_frames": 3},
            "pinch": {"motion": 0.06, "distance": 0.05, "stable_frames": 3}
        }
        
        # 动作检测参数
        self.initial_position = None  # 初始位置
        self.max_distance = 0  # 最大相对位移
        self.motion_buffer = deque(maxlen=100)  # 运动量缓冲区
        self.position_buffer = deque(maxlen=100)  # 位置缓冲区
        self.stable_frames = 0  # 静止帧计数器
        
        # 使用默认阈值初始化
        self.current_thresholds = self.gesture_thresholds["right_swipe"]
        print("动作检测参数初始化完成")

    def init_data_buffer(self):
        """初始化数据缓冲区"""
        self.sequence_length = SEQUENCE_LENGTH  # 使用全局常量
        self.data_buffer = deque(maxlen=self.sequence_length)
        self.current_gesture = "无动作"

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
        
        # 获取指尖坐标并计算相对位移
        # 索引6是食指尖(对应原始关键点8)
        index_finger = np.array(frame_data[6*3:6*3+2]) - palm  # 食指尖
        # 索引7是中指尖(对应原始关键点12)
        middle_finger = np.array(frame_data[7*3:7*3+2]) - palm  # 中指尖
        # 索引8是无名指尖(对应原始关键点16)
        ring_finger = np.array(frame_data[8*3:8*3+2]) - palm  # 无名指尖
        
        # 计算加权平均
        return 0.5 * index_finger + 0.25 * middle_finger + 0.25 * ring_finger

    def calculate_motion(self, current_data):
        """计算最近两帧之间的运动量"""
        if len(self.position_buffer) < 2:
            return 0
        prev_data = np.array(self.position_buffer[-2])
        curr_data = np.array(current_data)
        motion = np.mean(np.abs(curr_data - prev_data))
        return motion

    def load_gesture_names(self):
        """加载手势名称
        
        由于我们已经预定义了GESTURE_NAMES，此方法仅作为备用
        """
        # 直接返回预定义的手势名称
        return GESTURE_NAMES
        
        # 以下代码保留作为备用，以防需要从目录中读取
        # data_dir = "gesture_data"
        # return sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

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

        # 添加状态信息面板
        self.status_label = QLabel("状态: 未开始", self)
        self.status_label.setAlignment(Qt.AlignLeft)
        self.status_label.setStyleSheet("font-size: 16px; color: black;")
        
        self.threshold_label = QLabel("阈值信息: 未设置", self)
        self.threshold_label.setAlignment(Qt.AlignLeft)
        self.threshold_label.setStyleSheet("font-size: 16px; color: black;")
        
        self.confidence_label = QLabel("置信度: -", self)
        self.confidence_label.setAlignment(Qt.AlignLeft)
        self.confidence_label.setStyleSheet("font-size: 16px; color: black;")
        
        # 添加加载状态标签
        self.loading_label = QLabel("", self)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("font-size: 18px; color: blue; font-weight: bold;")

        # 布局
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.camera_id_input)
        control_layout.addWidget(self.start_stop_button)
        control_layout.addWidget(self.test_button)

        # 状态信息布局
        status_layout = QVBoxLayout()
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.threshold_label)
        status_layout.addWidget(self.confidence_label)
        
        # 将状态信息容器添加到一个水平布局
        info_container = QWidget()
        info_container.setLayout(status_layout)
        info_container.setFixedWidth(300)
        
        # 加载状态单独放在顶部
        loading_layout = QHBoxLayout()
        loading_layout.addWidget(self.loading_label)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.result_label, 2)
        bottom_layout.addWidget(info_container, 1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addLayout(loading_layout)
        main_layout.addWidget(self.preview_label)
        main_layout.addLayout(bottom_layout)

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
            model_complexity=0  
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def init_model(self):
        try:
            # 尝试先加载选定关键点的模型
            model_path = "saved_model_selected_landmarks"
            if not os.path.exists(model_path):
                model_path = "saved_model"  # 回退到原始模型
                print(f"警告: 找不到选定关键点的模型，尝试加载原始模型 {model_path}")
            
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
                print(f"使用关键点IDs: {SELECTED_LANDMARKS}")
                print(f"输入维度: {INPUT_DIM}")
                print(f"序列长度: {SEQUENCE_LENGTH}")
            except Exception as e:
                print(f"模型加载失败: {str(e)}")
                self.result_label.setText("模型加载失败")
                self.model = None
                return
            
            # 编译模型 - 与train.py中build_model使用的配置保持一致
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
                
                # 计算当前帧的运动量
                current_motion = self.calculate_motion(frame_data)
                self.motion_buffer.append(current_motion)
                self.position_buffer.append(frame_data)
                
                # 更新UI显示当前运动量
                self.update_motion_signal.emit("运动量", current_motion)
                
                # 计算相对位移
                if not self.action_started:
                    if current_motion > self.current_thresholds["motion"]:
                        self.action_started = True
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
                                    print(f"开始预测 (收集帧数: {len(self.data_buffer)})")
                                    self.update_action_signal.emit(f"开始预测 (帧数: {len(self.data_buffer)})", self.max_distance)
                                    self.run_prediction()
                            else:
                                print(f"动作不完整，放弃预测 (最大相对位移: {self.max_distance:.4f})")
                                self.update_action_signal.emit("动作不完整，放弃预测", self.max_distance)
                                self.action_started = False
                                self.stable_frames = 0
                                self.data_buffer.clear()
                    else:
                        self.stable_frames = 0
        else:
            if self.action_started:
                self.action_started = False
                self.data_buffer.clear()
            # 只有在未进行预测时才更新为无动作
            if not self.worker_thread or not self.worker_thread.isRunning():
                # 保留之前的预测结果，除非没有检测到手
                if not hasattr(self, 'current_gesture') or self.current_gesture == "无动作":
                    self.update_result_signal.emit("无动作")
            self.update_motion_signal.emit("无手势", 0.0)
            self.update_action_signal.emit("等待手势", 0.0)

        # 更新预览画面
        preview_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = preview_frame.shape
        q_img = QImage(preview_frame.data, w, h, w * c, QImage.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(q_img))

    def update_status_display(self, status_text):
        """更新加载状态显示"""
        self.loading_label.setText(status_text)
        # 确保UI立即更新
        QApplication.processEvents()
        
    def update_motion_display(self, text, value):
        """更新动作状态显示"""
        self.status_label.setText(f"状态: {text} ({value:.4f})")
        # 确保UI立即更新
        QApplication.processEvents()
        
    def update_action_display(self, text, value):
        """更新动作进度显示"""
        self.loading_label.setText(f"{text} - 最大位移: {value:.4f}")
        # 确保UI立即更新
        QApplication.processEvents()
        
    def run_prediction(self):
        try:
            # 显示加载状态
            self.update_status_signal.emit("正在准备数据...")
            
            # 动态填充/截断序列
            sequence = np.array(self.data_buffer)
            if len(sequence) < self.sequence_length:
                # 重复最后一帧填充不足部分 - 与train.py中load_dataset保持一致
                last_frame = sequence[-1]
                pad_frames = np.tile(last_frame, (self.sequence_length-len(sequence), 1))
                sequence = np.concatenate([sequence, pad_frames])
            else:
                sequence = sequence[-self.sequence_length:]  # 取最后30帧

            # 如果已有正在运行的线程，先停止它
            if self.worker is not None:
                self.worker.stop()
            if self.worker_thread is not None and self.worker_thread.isRunning():
                self.worker_thread.quit()
                self.worker_thread.wait(1000)  # 等待最多1秒

            # 更新UI状态
            self.update_status_signal.emit("初始化预测线程...")
            QApplication.processEvents()

            # 创建新的预测线程
            self.worker = PredictionWorker(self.model, sequence)
            self.worker_thread = QThread()

            # 设置连接
            self.worker.moveToThread(self.worker_thread)
            self.worker_thread.started.connect(self.worker.predict)
            self.worker.result_ready.connect(self.handle_prediction_result)
            self.worker.status_update.connect(self.update_status_signal.emit)
            self.worker.finished.connect(lambda: self.cleanup_thread())

            # 启动线程
            self.worker_thread.start()
            
        except Exception as e:
            print(f"创建预测线程时出错: {str(e)}")
            self.update_status_signal.emit(f"预测错误: {str(e)}")

    def cleanup_thread(self):
        try:
            # 清理完成后更新状态
            self.update_status_signal.emit("预测完成")
            
            if self.worker_thread is not None:
                self.worker_thread.quit()
                if not self.worker_thread.wait(1000):
                    self.worker_thread.terminate()
                self.worker_thread = None
            self.worker = None
            
            # 短暂延迟后清除加载状态
            QTimer.singleShot(2000, lambda: self.update_status_signal.emit(""))
            
        except Exception as e:
            print(f"清理线程时出错: {str(e)}")

    def handle_prediction_result(self, result, confidence):
        # 确保UI更新在主线程中执行
        QApplication.processEvents()
        
        # 更新置信度显示
        self.confidence_label.setText(f"置信度: {confidence:.4f}")
        
        # 更新当前手势的阈值
        if result in self.gesture_thresholds:
            self.current_thresholds = self.gesture_thresholds[result]
            print(f"更新阈值: {result}")
            print(f"运动量阈值: {self.current_thresholds['motion']}")
            print(f"位移阈值: {self.current_thresholds['distance']}")
            print(f"静止帧阈值: {self.current_thresholds['stable_frames']}")
            
            # 更新UI上的阈值信息
            thresh_info = (
                f"运动量阈值: {self.current_thresholds['motion']:.4f}\n"
                f"位移阈值: {self.current_thresholds['distance']:.4f}\n"
                f"静止帧阈值: {self.current_thresholds['stable_frames']}"
            )
            self.threshold_label.setText(thresh_info)
            
            # 立即更新状态显示
            self.update_motion_signal.emit(f"识别到: {result}", confidence)
        
        # 直接更新结果显示，不再使用无动作过滤
        if confidence > 0.2:  # 使用与PredictionWorker相同的阈值
            self.update_result_signal.emit(result)
            self.current_gesture = result
            self.last_prediction = result
            self.confirmation_count = 1
            # 立即更新可能的手势
            self.update_motion_signal.emit(f"检测到: {result}", confidence)
        else:
            self.update_result_signal.emit("无动作")
            self.update_motion_signal.emit("置信度过低", confidence)
        
        # 确保UI立即更新
        QApplication.processEvents()

    def update_result_display(self, text):
        self.result_label.setText(text)
        # 根据结果改变颜色
        color = "green" if text != "无动作" else "red"
        self.result_label.setStyleSheet(f"font-size: 24px; color: {color};")
        
        # 确保UI立即更新
        QApplication.processEvents()

    def test_model(self):
        """测试模型功能"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "模型未加载，请先加载模型")
            return

        try:
            # 加载测试数据
            test_data_path = "test_data.npy"
            if not os.path.exists(test_data_path):
                QMessageBox.warning(self, "警告", f"找不到测试数据文件 {test_data_path}")
                return
                
            test_data = np.load(test_data_path)
            
            # 检查是否需要提取关键点
            if test_data.shape[1] > INPUT_DIM:
                # 假设数据是完整的21个关键点，需要提取
                print("提取测试数据的选定关键点...")
                # 重塑为(帧数, 21, 3)以便处理
                frames = test_data.shape[0]
                reshaped_data = test_data.reshape(frames, 21, 3)
                # 提取选定关键点
                test_data = reshaped_data[:, SELECTED_LANDMARKS, :].reshape(frames, -1)
                print(f"提取后的测试数据形状: {test_data.shape}")
            
            if test_data.shape[0] < SEQUENCE_LENGTH:
                QMessageBox.warning(self, "警告", f"测试数据不足{SEQUENCE_LENGTH}帧，当前帧数: {test_data.shape[0]}")
                # 尝试填充数据
                last_frame = test_data[-1]
                pad_frames = np.tile(last_frame, (SEQUENCE_LENGTH - test_data.shape[0], 1))
                test_data = np.concatenate([test_data, pad_frames])
                print(f"已填充测试数据至{SEQUENCE_LENGTH}帧")
                
            elif test_data.shape[0] > SEQUENCE_LENGTH:
                # 截断为需要的帧数
                test_data = test_data[:SEQUENCE_LENGTH]
                print(f"已截断测试数据至{SEQUENCE_LENGTH}帧")

            # 进行预测
            input_data = np.expand_dims(test_data, axis=0)
            print(f"输入数据形状: {input_data.shape}")
            
            pred = self.model.predict(input_data, verbose=0)
            
            # 详细输出所有手势的置信度
            detailed_result = "各手势置信度:\n"
            for i, (gesture, conf) in enumerate(zip(GESTURE_NAMES, pred[0])):
                detailed_result += f"{gesture}: {conf:.4f}\n"
            
            gesture_idx = np.argmax(pred)
            confidence = pred[0][gesture_idx]
            
            result = f"预测结果: {GESTURE_NAMES[gesture_idx]}\n置信度: {confidence:.2%}\n\n{detailed_result}"
            QMessageBox.information(self, "测试结果", result)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "错误", f"测试过程中出错: {str(e)}\n\n{error_details}")

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