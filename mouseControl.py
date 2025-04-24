import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import math
import os
import threading
import time
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont

# 设置PyAutoGUI安全保护措施
pyautogui.FAILSAFE = False  # 禁用failsafe但添加自定义安全保护
# 设置移动速度和灵敏度
pyautogui.PAUSE = 0.003  # 降低最小操作间隔，提高响应速度，从0.005减少到0.003

class MouseControlThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)  # 用于发送处理后的帧
    status_signal = pyqtSignal(str)  # 用于发送状态信息

    def __init__(self, camera_id=1, screen_width=1920, screen_height=1080):
        super().__init__()
        # 设置屏幕尺寸
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # 获取实际屏幕尺寸
        try:
            screen_size = pyautogui.size()
            self.screen_width = screen_size.width
            self.screen_height = screen_size.height
            print(f"检测到屏幕分辨率: {self.screen_width}x{self.screen_height}")
        except Exception as e:
            print(f"获取屏幕分辨率出错: {str(e)}")
            print(f"使用默认值: {self.screen_width}x{self.screen_height}")
        
        # 添加控制状态标志
        self.is_control_active = True  # 控制是否激活
        
        # 安全区域设置
        self.safe_margin = 1/6  # 安全边距比例
        self.draw_safe_area = False  # 取消在画面上绘制安全区域指示框
        
        # 摄像头设置
        self.camera_id = camera_id
        self.running = False
        self.cap = None
        
        # 自定义安全区域 - 替代PyAutoGUI的FAILSAFE
        self.screen_safe_margin = 20  # 屏幕边缘安全区域（像素）
        self.previous_safe_error = 0  # 上次安全错误时间
        self.safe_error_cooldown = 2.0  # 安全错误提示冷却时间（秒）
        
        # 共享摄像头模式
        self.shared_camera_mode = True  # 默认使用共享模式
        self.shared_frame = None
        self.shared_frame_available = False
        self.shared_frame_lock = threading.Lock()
        
        # MediaPipe设置
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=0
        )
        
        # 鼠标控制状态
        self.prev_hand_center = None
        self.smoothing = 6  # 平滑因子，降低以提高响应速度
        self.is_left_clicked = False
        self.is_right_clicked = False
        self.is_scrolling = False
        self.scroll_mode = False
        self.prev_index_finger_y = None
        
        # 点击锁定位置 - 解决点击时位置漂移问题
        self.click_position_lock = False  # 是否锁定点击位置
        self.locked_mouse_position = None  # 锁定的鼠标位置
        self.click_lock_frames = 0  # 锁定计数器
        self.max_click_lock_frames = 5  # 最大锁定帧数，超过后允许拖拽
        
        # 点击稳定与精确控制
        self.pre_click_detection = False  # 检测到即将点击的状态
        self.pre_click_position = None    # 即将点击时的位置记录
        self.pre_click_count = 0          # 即将点击状态计数
        self.pre_click_threshold = 2      # 即将点击状态阈值
        self.palm_stable_threshold = 0.002  # 掌心稳定判定阈值
        self.palm_history = []            # 掌心位置历史
        self.palm_history_size = 5        # 掌心历史记录大小
        self.finger_palm_diff_threshold = 0.02  # 食指与掌心移动差异阈值
        self.last_palm_position = None    # 上一次掌心位置
        
        # 抖动抑制
        self.position_buffer = []  # 位置缓冲区用于平滑滤波
        self.buffer_size = 5  # 缩小位置缓冲区大小，平衡平滑度和响应性
        self.movement_threshold = 0.0005  # 移动阈值，提高灵敏度
        
        # 1D卡尔曼滤波器参数 - 用于每个单独的骨骼关键点坐标
        self.landmark_filter = {}  # 存储每个关键点的过滤状态
        
        # 额外的抖动抑制 - 卡尔曼滤波
        self.use_kalman_filter = True  # 是否使用卡尔曼滤波
        self.kalman_position_x = 0  # 卡尔曼滤波器估计的X位置
        self.kalman_position_y = 0  # 卡尔曼滤波器估计的Y位置
        self.kalman_process_noise = 0.03  # 过程噪声
        self.kalman_measurement_noise = 0.6  # 测量噪声
        self.kalman_initialized = False  # 卡尔曼滤波器是否已初始化
        
        # 滚动状态缓冲和控制
        self.scroll_buffer = []
        self.scroll_buffer_size = 3  # 调整为与之前代码一致
        self.scroll_speed = 0
        self.scroll_cooldown = 0
        self.last_scroll_time = 0
        self.scroll_threshold = 0.04  # 调整为与之前代码一致
        self.scroll_mode = False  # 是否处于滚轮模式
        self.prev_thumb_y = None  # 上一帧拇指Y坐标
        
        # 手指状态历史
        self.finger_history = []
        self.history_length = 3  # 调整为与之前代码一致
        
        # 节点距离阈值
        self.click_distance_threshold = 0.1  # 点击判定距离阈值
        self.scroll_trigger_threshold = 0.15  # 调整为与之前代码一致
        
        # 性能优化
        self.process_every_n_frames = 1  # 处理每一帧
        self.frame_count = 0
        
        # 添加缺失的帧缓存，用于UI显示
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # 减少帧处理的延迟
        self.last_process_time = 0
        self.min_process_interval = 0.01  # 处理间隔10ms
        
        # 添加手部检测状态
        self.hand_detected = False
        self.last_hand_detection_time = 0
        self.hand_lost_threshold = 0.2  # 手部丢失判定时间阈值(秒)
        
        print("鼠标控制模块初始化完成")

    def get_latest_frame(self):
        """获取最新的共享帧
        
        Returns:
            numpy.ndarray: 最新的帧图像，如果没有可用帧则返回None
        """
        if self.shared_frame_lock is not None:
            with self.shared_frame_lock:
                if self.shared_frame is not None and self.shared_frame_available:
                    return self.shared_frame.copy()
        return None

    def process_shared_frame(self, frame):
        """处理从主程序共享的帧"""
        try:
            if frame is not None:
                with self.shared_frame_lock:
                    self.shared_frame = frame.copy()  # 创建副本避免引用问题
                    self.shared_frame_available = True
        except Exception as e:
            print(f"处理共享帧出错: {str(e)}")
            
    def start_control(self, camera_id=None):
        """启动鼠标控制线程"""
        try:
            if camera_id is not None:
                self.camera_id = camera_id
            self.running = True
            
            print(f"开始鼠标控制线程, 摄像头ID: {self.camera_id}, 共享模式: {self.shared_camera_mode}")
            self.status_signal.emit("鼠标控制已启动")
            
            if not self.isRunning():
                self.start()
                print("鼠标控制线程已启动")
            else:
                print("线程已经在运行")
        except Exception as e:
            print(f"启动鼠标控制出错: {str(e)}")
            self.status_signal.emit(f"启动出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def stop_control(self):
        """停止鼠标控制线程"""
        try:
            print("停止鼠标控制线程")
            self.running = False
            self.status_signal.emit("停止中...")
            
            # 确保释放鼠标按键
            if self.is_left_clicked:
                pyautogui.mouseUp(button='left')
                self.is_left_clicked = False
                
            if self.is_right_clicked:
                pyautogui.mouseUp(button='right')
                self.is_right_clicked = False
            
            # 清理共享帧
            with self.shared_frame_lock:
                self.shared_frame = None
                self.shared_frame_available = False
                
            # 释放摄像头资源(如果不是共享模式)
            if not self.shared_camera_mode and self.cap and self.cap.isOpened():
                print("释放摄像头资源")
                self.cap.release()
                self.cap = None
                
            self.status_signal.emit("已停止")
            print("鼠标控制已停止")
        except Exception as e:
            print(f"停止鼠标控制出错: {str(e)}")
            self.status_signal.emit(f"停止出错: {str(e)}")

    def calculate_distance(self, p1, p2):
        """计算两点之间的欧几里得距离"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def is_touching(self, p1, p2, threshold=None):
        """判断两个关键点是否接触"""
        if threshold is None:
            threshold = self.click_distance_threshold
        return self.calculate_distance(p1, p2) < threshold

    def check_scroll_gesture(self, landmarks):
        """检查是否是滚动手势
        
        滚动手势的判定：
        - 拇指(4)与无名指尖(16)接触 - 使用无名指
        - 向上或向下移动手指可以控制滚动方向和速度
        """
        # 获取关键点
        thumb_tip = landmarks.landmark[4]  # 拇指尖
        ring_finger_tip = landmarks.landmark[16]  # 无名指尖 - 改回无名指
        
        # 检查拇指与无名指是否触碰
        trigger_threshold = self.scroll_trigger_threshold * 1.2  # 适当调整阈值
        is_touching_ring = self.is_touching(thumb_tip, ring_finger_tip, trigger_threshold)
        is_scroll_gesture = is_touching_ring
        
        # 更新历史记录
        self.finger_history.append(is_scroll_gesture)
        if len(self.finger_history) > self.history_length:
            self.finger_history.pop(0)
            
        # 判断是否稳定地进入滚动模式
        stable_scroll = False
        if len(self.finger_history) >= 2:
            # 只需最近帧有一帧满足条件即可
            stable_scroll = sum(self.finger_history[-2:]) >= 1
            
        # 计算滚动速度和方向
        if stable_scroll or is_scroll_gesture:
            # 确保从触摸开始时重置垂直位置
            if not self.scroll_mode:
                self.prev_thumb_y = thumb_tip.y
                self.scroll_mode = True
                self.scroll_buffer = []
            
            # 计算垂直移动
            if self.prev_thumb_y is not None:
                # 计算相对于上一帧的移动距离，直接使用拇指的垂直位移
                scroll_amount = (thumb_tip.y - self.prev_thumb_y) * 250  # 增大倍数，提高灵敏度
                
                # 添加到缓冲区
                self.scroll_buffer.append(scroll_amount)
                if len(self.scroll_buffer) > self.scroll_buffer_size:
                    self.scroll_buffer.pop(0)
                    
                # 平均滚动量，减少抖动
                if len(self.scroll_buffer) > 0:
                    avg_scroll = sum(self.scroll_buffer) / len(self.scroll_buffer)
                    if abs(avg_scroll) > 0.04:  # 减小死区，提高灵敏度
                        # 向下移动拇指，屏幕向下滚动(正值)
                        scroll_clicks = int(avg_scroll * 12)  # 倍数调整
                        if scroll_clicks != 0:
                            pyautogui.scroll(scroll_clicks)
                            self.last_scroll_time = time.time()
                            print(f"执行滚动: {scroll_clicks}")  # 添加调试信息
            
            # 更新拇指位置
            self.prev_thumb_y = thumb_tip.y
            return True
        else:
            # 重置滚动状态
            self.scroll_mode = False
            self.scroll_buffer = []
            self.prev_thumb_y = None
            return False

    def draw_control_point(self, frame, landmark, color, label):
        """在帧上标记控制点"""
        h, w, _ = frame.shape
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        # 绘制圆圈
        cv2.circle(frame, (cx, cy), 10, color, -1)
        # 不再使用cv2.putText添加标签，在主循环中通过PIL绘制
        # 这里只返回坐标和标签信息
        return (cx, cy, label, color)
        
    def map_to_screen(self, x, y, frame_width, frame_height):
        """将摄像头坐标映射到屏幕坐标，考虑安全区域
        
        Args:
            x, y: 原始坐标 (0-1范围)
            frame_width, frame_height: 帧的宽度和高度
            
        Returns:
            映射后的屏幕坐标 (像素)
        """
        # 计算安全区域的范围
        safe_x_min = self.safe_margin
        safe_x_max = 1.0 - self.safe_margin
        safe_y_min = self.safe_margin
        safe_y_max = 1.0 - self.safe_margin
        
        # 使用快速线性映射方式，减少条件判断
        # 将输入值限制在安全区域范围内
        x = max(safe_x_min, min(safe_x_max, x))
        y = max(safe_y_min, min(safe_y_max, y))
        
        # 归一化到[0,1]范围
        normalized_x = (x - safe_x_min) / (safe_x_max - safe_x_min)
        normalized_y = (y - safe_y_min) / (safe_y_max - safe_y_min)
        
        # 映射到屏幕坐标
        screen_x = int(normalized_x * self.screen_width)
        screen_y = int(normalized_y * self.screen_height)
        
        return screen_x, screen_y

    def _filter_landmark_coordinates(self, landmarks):
        """对手部关键点坐标应用1D卡尔曼滤波，减少抖动"""
        filtered_landmarks = []
        
        for i, landmark in enumerate(landmarks.landmark):
            # 每个关键点独立应用卡尔曼滤波
            if i not in self.landmark_filter:
                # 初始化该关键点的滤波器状态
                self.landmark_filter[i] = {
                    'x': {'pos': landmark.x, 'vel': 0, 'initialized': True},
                    'y': {'pos': landmark.y, 'vel': 0, 'initialized': True},
                    'z': {'pos': landmark.z, 'vel': 0, 'initialized': True}
                }
                filtered_landmarks.append(landmark)
                continue
            
            # 创建新的关键点对象
            filtered_landmark = type('obj', (object,), {
                'x': 0, 'y': 0, 'z': 0
            })
            
            # 为每个坐标分量应用滤波
            for coord in ['x', 'y', 'z']:
                # 获取原始值
                orig_value = getattr(landmark, coord)
                
                # 获取滤波器状态
                filter_state = self.landmark_filter[i][coord]
                
                if not filter_state['initialized']:
                    # 首次初始化
                    filter_state['pos'] = orig_value
                    filter_state['vel'] = 0
                    filter_state['initialized'] = True
                    setattr(filtered_landmark, coord, orig_value)
                    continue
                
                # 更新位置 - 简化版，不使用速度
                filter_state['pos'] = filter_state['pos'] * 0.6 + orig_value * 0.4
                
                # 设置过滤后的值
                setattr(filtered_landmark, coord, filter_state['pos'])
            
            filtered_landmarks.append(filtered_landmark)
        
        # 创建一个新的landmarks对象，包含过滤后的关键点
        filtered_result = type('obj', (object,), {'landmark': filtered_landmarks})
        return filtered_result

    def apply_position_smoothing(self, x, y):
        """应用简单平滑处理减少抖动"""
        # 添加当前位置到缓冲区
        self.position_buffer.append((x, y))
        
        # 限制缓冲区大小
        if len(self.position_buffer) > self.buffer_size:
            self.position_buffer.pop(0)
        
        # 如果缓冲区还不够大，直接返回当前值
        if len(self.position_buffer) < 3:
            return x, y
        
        # 简单的移动平均平滑
        avg_x = sum(p[0] for p in self.position_buffer) / len(self.position_buffer)
        avg_y = sum(p[1] for p in self.position_buffer) / len(self.position_buffer)
        
        # 应用卡尔曼滤波进一步平滑
        if self.use_kalman_filter:
            avg_x, avg_y = self._apply_kalman_filter(avg_x, avg_y)
        
        # 确保在屏幕范围内
        x = max(self.screen_safe_margin, min(int(avg_x), self.screen_width - self.screen_safe_margin))
        y = max(self.screen_safe_margin, min(int(avg_y), self.screen_height - self.screen_safe_margin))
        
        return x, y

    def _apply_kalman_filter(self, x, y):
        """应用卡尔曼滤波器平滑鼠标位置 - 简化版，无速度估计"""
        # 如果是首次调用，初始化卡尔曼滤波器状态
        if not self.kalman_initialized:
            self.kalman_position_x = x
            self.kalman_position_y = y
            self.kalman_initialized = True
            return x, y
        
        # 1. 预测步骤 - 简化，只使用位置
        predicted_x = self.kalman_position_x
        predicted_y = self.kalman_position_y
        
        # 2. 更新步骤
        # 计算测量值与预测值的差异
        residual_x = x - predicted_x
        residual_y = y - predicted_y
        
        # 卡尔曼增益 (0到1之间的值)
        kalman_gain = self.kalman_process_noise / (self.kalman_process_noise + self.kalman_measurement_noise)
        
        # 更新位置估计
        self.kalman_position_x = predicted_x + kalman_gain * residual_x
        self.kalman_position_y = predicted_y + kalman_gain * residual_y
        
        return int(self.kalman_position_x), int(self.kalman_position_y)

    def check_safe_position(self, x, y):
        """自定义安全位置检查，替代PyAutoGUI的failsafe
        
        Args:
            x, y: 鼠标坐标
            
        Returns:
            安全的鼠标坐标
        """
        # 检查是否在屏幕边缘安全区域内
        margin = self.screen_safe_margin
        
        # 限制坐标在安全范围内
        safe_x = max(margin, min(self.screen_width - margin, x))
        safe_y = max(margin, min(self.screen_height - margin, y))
        
        # 如果有调整，且冷却时间已过，输出警告
        if (safe_x != x or safe_y != y) and time.time() - self.previous_safe_error > self.safe_error_cooldown:
            self.previous_safe_error = time.time()
            # 通过状态信号显示警告而不是打印
            self.status_signal.emit("已阻止鼠标移动到屏幕边缘")
            
        return safe_x, safe_y
        
    def safe_mouse_move(self, x, y):
        """安全的鼠标移动，包含错误处理和位置检查"""
        try:
            # 应用自定义安全检查
            safe_x, safe_y = self.check_safe_position(x, y)
            pyautogui.moveTo(safe_x, safe_y)
            return True
        except Exception as e:
            # 减少控制台输出，使用状态信号
            if time.time() - self.previous_safe_error > self.safe_error_cooldown:
                self.previous_safe_error = time.time()
                self.status_signal.emit(f"鼠标移动错误: {str(e)[:50]}...")
            return False

    def run(self):
        """运行鼠标控制线程，处理手部跟踪和鼠标动作"""
        import traceback
        try:
            print("启动鼠标控制线程...")
            self.status_signal.emit("鼠标控制已启动")
            
            # 最小帧间隔（秒），理论上支持120FPS
            min_frame_interval = 0.008
            last_process_time = time.time()
            
            # 检查摄像头是否可用
            if not self.shared_camera_mode:
                try:
                    self.cap = cv2.VideoCapture(self.camera_id)
                    if not self.cap.isOpened():
                        self.status_signal.emit("无法访问摄像头")
                        return
                        
                    # 设置摄像头参数 - 使用更高效的配置
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 60)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 设置缓冲区大小为1，减少延迟
                except Exception as e:
                    self.status_signal.emit(f"摄像头初始化错误: {str(e)}")
                    return
            
            # 初始化MediaPipe Hands - 使用性能优先配置
            hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=0  # 性能优先
            )
            
            mp_drawing = mp.solutions.hands
            prev_hand_landmark = None
            self.running = True
            
            # 性能监控变量
            frame_count = 0
            process_count = 0
            fps_start_time = time.time()
            current_fps = 0
            process_fps = 0
            
            # 处理时间跟踪
            processing_times = []
            MAX_TIME_SAMPLES = 30  # 保留最近30个处理时间样本
            
            # 帧跳过计数器
            frame_skip_counter = 0
            
            while self.running:
                current_time = time.time()
                
                # 控制处理频率，避免过度处理
                elapsed = current_time - last_process_time
                if elapsed < min_frame_interval:
                    # 精确休眠以达到目标帧率
                    time.sleep(max(0.001, min_frame_interval - elapsed - 0.001))
                    continue
                
                # 获取帧 - 优先使用共享模式，减少代码重复
                frame = None
                if self.shared_camera_mode:
                    with self.shared_frame_lock:
                        if self.shared_frame is not None and self.shared_frame_available:
                            frame = self.shared_frame.copy()
                elif self.cap is not None and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        # 无效帧，短暂休眠后继续
                        time.sleep(0.001)
                        continue
                
                # 无帧可处理，继续下一轮
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                # 更新总帧率计数器
                frame_count += 1
                
                # FPS计算 - 每秒更新一次
                if current_time - fps_start_time >= 1.0:
                    current_fps = frame_count
                    process_fps = process_count
                    
                    # 计算平均处理时间
                    if processing_times:
                        avg_time = sum(processing_times) / len(processing_times)
                        print(f"鼠标控制 总FPS: {current_fps}, 处理FPS: {process_fps}, 平均处理时间: {avg_time*1000:.1f}ms")
                    else:
                        print(f"鼠标控制 总FPS: {current_fps}, 处理FPS: {process_fps}")
                    
                    frame_count = 0
                    process_count = 0
                    fps_start_time = current_time
                    
                # 自适应帧处理 - 在高帧率时适当跳过帧
                # 仅当控制激活且上一帧处理时间过长时应用
                should_process = True
                if self.is_control_active and processing_times and len(processing_times) > 5:
                    avg_process_time = sum(processing_times[-5:]) / 5
                    target_process_rate = 30  # 目标处理帧率
                    
                    # 如果处理时间超过目标帧率的时间，考虑跳过帧
                    if avg_process_time > 1.0 / target_process_rate:
                        frame_skip_counter += 1
                        if frame_skip_counter < 2:  # 最多跳过一帧
                            should_process = False
                    else:
                        frame_skip_counter = 0
                else:
                    frame_skip_counter = 0
                
                # 如果决定跳过此帧处理，直接更新UI并继续
                if not should_process:
                    # 如果需要显示UI，直接传递原始帧
                    if hasattr(self, 'frame_signal') and self.is_control_active:
                        # 复制帧用于绘制
                        display_frame = frame.copy()
                        
                        # 仅绘制手部连接线，不添加文字
                        self.frame_signal.emit(display_frame)
                    last_process_time = time.time()
                    continue
                
                # 记录处理开始时间
                process_start = time.time()
                
                # 仅在控制激活时处理手部检测
                if self.is_control_active:
                    # 为UI准备显示帧（仅当需要时）
                    need_display = hasattr(self, 'frame_signal')
                    display_frame = frame.copy() if need_display else None
                    
                    # 优化：将BGR->RGB转换与MediaPipe处理合并
                    # MediaPipe可直接处理BGR图像
                    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    # 手部检测标志
                    hand_detected = False
                    
                    # 处理检测结果
                    if results.multi_hand_landmarks:
                        hand_landmark = results.multi_hand_landmarks[0]
                        hand_detected = True
                        self.last_hand_detection_time = current_time
                        self.hand_detected = True
                        
                        # 仅当需要显示UI时绘制手部标记
                        if need_display and display_frame is not None:
                            mp.solutions.drawing_utils.draw_landmarks(
                                display_frame, 
                                hand_landmark, 
                                mp.solutions.hands.HAND_CONNECTIONS
                            )
                        
                        # 应用手部关键点滤波减少抖动
                        filtered_hand_landmark = self._filter_landmark_coordinates(hand_landmark)
                        
                        # 计算掌心位置
                        palm_center = self._calculate_palm_center(filtered_hand_landmark)
                        
                        # 获取手指状态
                        fingers_state = self._get_fingers_state(filtered_hand_landmark)
                        
                        # 处理鼠标动作
                        self._process_mouse_actions(palm_center, fingers_state, filtered_hand_landmark)
                        
                        prev_hand_landmark = filtered_hand_landmark
                    else:
                        # 重置鼠标状态（仅当需要时）
                        if self.is_left_clicked or self.is_right_clicked:
                            if self.is_left_clicked:
                                pyautogui.mouseUp(button='left')
                                self.is_left_clicked = False
                            if self.is_right_clicked:
                                pyautogui.mouseUp(button='right')
                                self.is_right_clicked = False
                            self.is_dragging = False
                        
                        prev_hand_landmark = None
                        self.hand_detected = False
                    
                    # 更新UI显示，但不绘制文字（仅当需要时）
                    if need_display and display_frame is not None:
                        # 发送处理后的帧，不添加额外文字
                        self.frame_signal.emit(display_frame)
                elif hasattr(self, 'frame_signal'):
                    # 控制不活跃时，直接传递原始帧，不添加文字
                    self.frame_signal.emit(frame)
                
                # 记录处理时间并更新处理帧计数
                process_time = time.time() - process_start
                processing_times.append(process_time)
                process_count += 1
                if len(processing_times) > MAX_TIME_SAMPLES:
                    processing_times.pop(0)
                
                # 高级动态调整 - 自适应目标帧率
                # 根据处理时间自动调整目标帧率
                if processing_times and len(processing_times) > 5:
                    recent_avg = sum(processing_times[-5:]) / 5
                    
                    # 动态目标帧率 - 根据处理能力调整
                    if recent_avg < 0.01:  # 处理速度快 - 提高目标
                        target_fps = 60
                    elif recent_avg < 0.02:  # 中等处理速度
                        target_fps = 45
                    else:  # 处理速度慢
                        target_fps = 30
                    
                    target_interval = 1.0 / target_fps
                else:
                    # 默认目标30FPS
                    target_interval = 1.0 / 30.0
                
                # 计算需要的睡眠时间，确保达到但不超过目标帧率
                if process_time < target_interval:
                    sleep_time = target_interval - process_time
                    # 考虑系统调度开销，略微减少睡眠时间
                    adjusted_sleep = max(0.001, sleep_time * 0.95)
                    time.sleep(adjusted_sleep)
                
                # 更新时间戳
                last_process_time = time.time()
            
            # 线程结束，清理资源
            print("鼠标控制线程结束")
            if hasattr(self, 'status_signal'):
                self.status_signal.emit("鼠标控制已停止")
            
            # 重置鼠标状态
            if self.is_left_clicked:
                pyautogui.mouseUp(button='left')
                self.is_left_clicked = False
            if self.is_right_clicked:
                pyautogui.mouseUp(button='right')
                self.is_right_clicked = False
            self.is_dragging = False
            
            # 释放资源
            hands.close()
            
            # 释放摄像头
            if not self.shared_camera_mode and self.cap is not None and self.cap.isOpened():
                self.cap.release()
                
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"鼠标控制线程错误: {str(e)}\n{traceback_str}")
            if hasattr(self, 'status_signal'):
                self.status_signal.emit(f"鼠标控制错误: {str(e)}")
            
            # 确保资源释放
            if not self.shared_camera_mode and hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            
            # 重置鼠标状态
            if hasattr(self, 'is_left_clicked') and self.is_left_clicked:
                pyautogui.mouseUp(button='left')
                self.is_left_clicked = False
            if hasattr(self, 'is_right_clicked') and self.is_right_clicked:
                pyautogui.mouseUp(button='right')
                self.is_right_clicked = False
            self.is_dragging = False

    def _calculate_palm_center(self, hand_landmark):
        """计算掌心中心点的位置
        
        参数:
            hand_landmark: 手部关键点坐标
            
        返回:
            掌心中心点坐标
        """
        # 使用改进的掌心计算方法
        wrist = hand_landmark.landmark[0]  # 手腕点
        middle_base = hand_landmark.landmark[9]  # 中指根部
        ring_base = hand_landmark.landmark[13]  # 无名指根部
        
        # 使用手腕点和中指根部的加权平均计算掌心
        # 这提供了更稳定的掌心定位
        palm_x = wrist.x * 0.4 + middle_base.x * 0.3 + ring_base.x * 0.3
        palm_y = wrist.y * 0.4 + middle_base.y * 0.3 + ring_base.y * 0.3
        palm_z = wrist.z * 0.4 + middle_base.z * 0.3 + ring_base.z * 0.3
        
        # 创建掌心坐标点
        class PalmPoint:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
                
        return PalmPoint(palm_x, palm_y, palm_z)
        
    def _get_fingers_state(self, hand_landmark):
        """获取手指状态（是否伸直）"""
        # 拇指：比较拇指尖与第一关节的x坐标
        thumb_is_open = hand_landmark.landmark[4].x > hand_landmark.landmark[3].x
        
        # 其他手指：比较指尖与掌心的y坐标差值
        # 如果指尖的y坐标小于对应指根的y坐标，则认为手指伸直
        index_is_open = hand_landmark.landmark[8].y < hand_landmark.landmark[5].y
        middle_is_open = hand_landmark.landmark[12].y < hand_landmark.landmark[9].y
        ring_is_open = hand_landmark.landmark[16].y < hand_landmark.landmark[13].y
        pinky_is_open = hand_landmark.landmark[20].y < hand_landmark.landmark[17].y
        
        return [thumb_is_open, index_is_open, middle_is_open, ring_is_open, pinky_is_open]

    def _process_mouse_actions(self, palm_center, fingers_state, prev_hand_landmark):
        """处理鼠标动作逻辑"""
        try:
            # 检查是否进入滚动模式
            is_scrolling = False
            if prev_hand_landmark is not None:
                is_scrolling = self.check_scroll_gesture(prev_hand_landmark)
                
            # 如果在滚动模式中，不处理其他鼠标动作
            if is_scrolling:
                self.is_scrolling = True
                # 输出滚动状态
                self.status_signal.emit("滚轮模式")
                return
            else:
                self.is_scrolling = False
                if self.scroll_mode:  # 如果刚退出滚轮模式，重置相关状态
                    self.scroll_mode = False
                    self.scroll_buffer = []
                    self.prev_thumb_y = None
            
            # 获取手部关键点，用于手势识别和点击稳定性
            if prev_hand_landmark is not None:
                thumb_tip = prev_hand_landmark.landmark[4]  # 拇指尖
                index_finger_tip = prev_hand_landmark.landmark[8]  # 食指尖
                middle_finger_tip = prev_hand_landmark.landmark[12]  # 中指尖
                
                # 记录掌心位置历史 - 用于判断掌心稳定性
                if palm_center is not None:
                    self.palm_history.append((palm_center.x, palm_center.y))
                    if len(self.palm_history) > self.palm_history_size:
                        self.palm_history.pop(0)
                
                # 计算掌心稳定性
                palm_stable = False
                if len(self.palm_history) >= 3:
                    max_palm_movement = 0
                    for i in range(1, len(self.palm_history)):
                        px_diff = abs(self.palm_history[i][0] - self.palm_history[i-1][0])
                        py_diff = abs(self.palm_history[i][1] - self.palm_history[i-1][1])
                        palm_movement = math.sqrt(px_diff*px_diff + py_diff*py_diff)
                        max_palm_movement = max(max_palm_movement, palm_movement)
                    
                    # 判断掌心是否稳定
                    palm_stable = max_palm_movement < self.palm_stable_threshold
                
                # 检测食指与拇指是否接近 - 用于提前进入预点击状态
                distance = self.calculate_distance(thumb_tip, index_finger_tip)
                approaching_click = distance < self.click_distance_threshold * 2  # 距离是触碰阈值的2倍时进入预点击
                
                # 检测食指与掌心的相对移动
                finger_moved_more_than_palm = False
                if self.last_palm_position is not None:
                    # 掌心移动量
                    palm_dx = abs(palm_center.x - self.last_palm_position[0])
                    palm_dy = abs(palm_center.y - self.last_palm_position[1])
                    palm_movement = math.sqrt(palm_dx*palm_dx + palm_dy*palm_dy)
                    
                    # 食指移动量(相对于上一帧)
                    if hasattr(self, 'last_index_position') and self.last_index_position is not None:
                        index_dx = abs(index_finger_tip.x - self.last_index_position[0])
                        index_dy = abs(index_finger_tip.y - self.last_index_position[1])
                        index_movement = math.sqrt(index_dx*index_dx + index_dy*index_dy)
                        
                        # 食指移动明显大于掌心移动时
                        finger_moved_more_than_palm = (index_movement > palm_movement + self.finger_palm_diff_threshold)
                
                # 更新上一帧的位置记录
                self.last_palm_position = (palm_center.x, palm_center.y)
                self.last_index_position = (index_finger_tip.x, index_finger_tip.y)
                
                # 预点击状态逻辑
                if approaching_click and not self.is_left_clicked and palm_stable:
                    if not self.pre_click_detection:
                        self.pre_click_detection = True
                        self.pre_click_position = self.last_mouse_pos if self.last_mouse_pos else None
                        self.pre_click_count = 0
                    else:
                        self.pre_click_count += 1
                else:
                    # 如果不再满足预点击条件，重置状态
                    if not (approaching_click and palm_stable):
                        self.pre_click_detection = False
                        self.pre_click_position = None
                        self.pre_click_count = 0
                
                # 检测左击和右击
                is_left_touching = self.is_touching(thumb_tip, index_finger_tip)
                is_right_touching = self.is_touching(thumb_tip, middle_finger_tip)
                
                # 左键点击逻辑 - 加入预点击位置锁定
                if is_left_touching and not is_scrolling:
                    if not self.is_left_clicked:
                        # 新增: 使用预点击位置而不是当前位置
                        if self.pre_click_detection and self.pre_click_position and self.pre_click_count > self.pre_click_threshold:
                            # 使用预先记录的位置进行点击
                            click_x, click_y = self.pre_click_position
                            pyautogui.moveTo(click_x, click_y, duration=0.0, _pause=False)
                            pyautogui.mouseDown(button='left')
                            self.is_left_clicked = True
                            self.locked_mouse_position = self.pre_click_position
                            self.click_position_lock = True
                            self.click_lock_frames = 0  # 重置锁定帧计数
                        else:
                            # 常规点击 - 锁定当前位置
                            if self.last_mouse_pos:
                                pyautogui.mouseDown(button='left')
                                self.is_left_clicked = True
                                self.locked_mouse_position = self.last_mouse_pos
                                self.click_position_lock = True
                                self.click_lock_frames = 0  # 重置锁定帧计数
                    
                    # 如果拇指和食指保持接触，则进入拖拽模式
                    self.is_dragging = True
                    
                    # 在拖拽时，处理拖拽逻辑
                    if self.is_dragging:
                        # 点击持续中，处理锁定和拖拽
                        if self.locked_mouse_position:
                            # 点击保持一段时间后允许拖动
                            self.click_lock_frames += 1
                            if self.click_lock_frames > self.max_click_lock_frames:
                                # 超过最大锁定帧数，允许拖动
                                self.click_position_lock = False
                                self.status_signal.emit("拖动模式")
                            else:
                                # 仍在锁定状态，保持鼠标位置
                                if self.last_mouse_pos != self.locked_mouse_position:
                                    pyautogui.moveTo(self.locked_mouse_position[0], self.locked_mouse_position[1], duration=0.0, _pause=False)
                                    self.last_mouse_pos = self.locked_mouse_position
                elif is_right_touching and not is_scrolling and not self.is_left_clicked:
                    # 右键点击 - 只有在没有左键点击时才处理右键
                    if not self.is_right_clicked:
                        # 新增: 使用预点击位置或当前位置
                        if self.pre_click_detection and self.pre_click_position and self.pre_click_count > self.pre_click_threshold:
                            # 使用预先记录的位置进行点击
                            click_x, click_y = self.pre_click_position
                            pyautogui.moveTo(click_x, click_y, duration=0.0, _pause=False)
                        pyautogui.mouseDown(button='right')
                        self.is_right_clicked = True
                else:
                    # 结束所有点击
                    if self.is_left_clicked:
                        pyautogui.mouseUp(button='left')
                        self.is_left_clicked = False
                        self.is_dragging = False
                        self.click_position_lock = False
                        self.locked_mouse_position = None
                    
                    if self.is_right_clicked:
                        pyautogui.mouseUp(button='right')
                        self.is_right_clicked = False
            
            # 处理实际鼠标移动 - 仅当不处于点击锁定状态时
            if self.is_control_active and not self.click_position_lock:
                # 获取掌心的规范化坐标(0-1范围)
                palm_x, palm_y = palm_center.x, palm_center.y
                
                # 获取食指尖位置，用于计算偏移量
                index_finger_tip = prev_hand_landmark.landmark[8]  # 食指尖
                
                # 计算从掌心到食指尖的方向向量
                direction_x = index_finger_tip.x - palm_x
                direction_y = index_finger_tip.y - palm_y
                
                # 应用一定比例的偏移，保持控制稳定性的同时将鼠标指针移向食指尖
                # 偏移系数为0.6，保留一部分掌心控制的稳定性，同时向食指尖方向偏移
                offset_ratio = 0.6
                adjusted_x = palm_x + direction_x * offset_ratio
                adjusted_y = palm_y + direction_y * offset_ratio
                
                # 直接映射到屏幕坐标，使用优化的map_to_screen方法
                h, w = 1, 1  # 规范化坐标使用单位高宽
                screen_x, screen_y = self.map_to_screen(adjusted_x, adjusted_y, w, h)
                
                # 使用简单平滑
                if len(self.position_buffer) >= self.buffer_size:
                    self.position_buffer.pop(0)
                self.position_buffer.append((screen_x, screen_y))
                
                smoothed_x, smoothed_y = self.apply_position_smoothing(screen_x, screen_y)
                
                # 直接移动鼠标，不判断是否需要移动
                pyautogui.moveTo(smoothed_x, smoothed_y, duration=0.0, _pause=False)
                self.last_mouse_pos = (smoothed_x, smoothed_y)
                
            elif self.click_position_lock and self.locked_mouse_position:
                # 如果处于点击锁定状态，强制保持鼠标在锁定位置
                lock_x, lock_y = self.locked_mouse_position
                if self.last_mouse_pos != self.locked_mouse_position:
                    pyautogui.moveTo(lock_x, lock_y, duration=0.0, _pause=False)
                    self.last_mouse_pos = self.locked_mouse_position
                
        except Exception as e:
            import traceback
            print(f"处理鼠标动作时出错: {str(e)}")
            traceback.print_exc()

    def _is_in_safe_area(self, x, y, frame_width, frame_height):
        """判断坐标是否在安全区域内
        
        参数:
            x, y: 坐标点(像素)
            frame_width, frame_height: 帧宽高
            
        返回:
            布尔值，表示是否在安全区域内
        """
        # 计算安全区域边界
        margin_x = int(frame_width * self.safe_margin)
        margin_y = int(frame_height * self.safe_margin)
        
        # 判断点是否在安全区域内
        return (margin_x <= x <= frame_width - margin_x) and (margin_y <= y <= frame_height - margin_y)

    def _map_value(self, value, in_min, in_max, out_min, out_max):
        """将一个值从一个范围映射到另一个范围
        
        优化的映射算法，减少计算量
        
        Args:
            value: 要映射的值
            in_min, in_max: 输入范围
            out_min, out_max: 输出范围
            
        Returns:
            映射后的值
        """
        # 防止除以零错误
        if in_max == in_min:
            return out_min
        
        # 确保值在输入范围内
        value = max(in_min, min(in_max, value))
        
        # 快速线性映射计算
        return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)

# 如果直接运行此文件，执行简单的测试
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
    
    app = QApplication([])
    window = QMainWindow()
    window.setWindowTitle("鼠标控制测试")
    window.setGeometry(100, 100, 800, 600)
    
    label = QLabel("按下按钮开始控制鼠标")
    label.setAlignment(Qt.AlignCenter)
    label.setFont(QFont("微软雅黑", 14))
    
    preview = QLabel()
    preview.setFixedSize(640, 480)
    preview.setStyleSheet("background-color: black;")
    
    start_button = QPushButton("开始控制")
    stop_button = QPushButton("停止控制")
    stop_button.setEnabled(False)
    
    layout = QVBoxLayout()
    layout.addWidget(label)
    layout.addWidget(preview)
    layout.addWidget(start_button)
    layout.addWidget(stop_button)
    
    container = QWidget()
    container.setLayout(layout)
    window.setCentralWidget(container)
    
    mouse_control = MouseControlThread()
    
    def update_frame(frame):
        h, w, c = frame.shape
        bytes_per_line = w * c
        from PyQt5.QtGui import QImage, QPixmap
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        preview.setPixmap(QPixmap.fromImage(q_img))
    
    def update_status(status):
        label.setText(status)
    
    def start_control():
        mouse_control.start_control()
        start_button.setEnabled(False)
        stop_button.setEnabled(True)
    
    def stop_control():
        mouse_control.stop_control()
        start_button.setEnabled(True)
        stop_button.setEnabled(False)
    
    mouse_control.frame_signal.connect(update_frame)
    mouse_control.status_signal.connect(update_status)
    start_button.clicked.connect(start_control)
    stop_button.clicked.connect(stop_control)
    
    window.show()
    
    # 确保应用关闭时停止线程
    app.aboutToQuit.connect(mouse_control.stop_control)
    
    sys.exit(app.exec_())
