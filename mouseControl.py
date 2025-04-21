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
pyautogui.PAUSE = 0.005  # 降低最小操作间隔，提高响应速度，从0.01减少到0.005

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
        
        # 安全区域设置 - 摄像头画面上下左右各留出1/8的区域，减小边缘区域
        self.safe_margin = 1/8  # 安全边距比例，从1/6改为1/8
        self.draw_safe_area = True  # 是否在画面上绘制安全区域指示框
        
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
        self.smoothing = 6  # 减少平滑因子，从8减少到6，增加响应速度
        self.is_left_clicked = False
        self.is_right_clicked = False
        self.is_scrolling = False
        self.scroll_mode = False
        self.prev_index_finger_y = None
        
        # 点击锁定位置 - 解决点击时位置漂移问题
        self.click_position_lock = False  # 是否锁定点击位置
        self.locked_mouse_position = None  # 锁定的鼠标位置
        self.click_lock_frames = 0  # 锁定计数器
        self.max_click_lock_frames = 5  # 最大锁定帧数
        
        # 抖动抑制
        self.position_buffer = []  # 位置缓冲区用于平滑滤波
        self.buffer_size = 8  # 位置缓冲区大小，从10减少到8
        self.movement_threshold = 0.0015  # 移动阈值，从0.002改为0.0015
        self.is_stable = False  # 静止状态标志
        self.stable_count = 0  # 静止帧计数
        self.stable_threshold = 2  # 静止判定阈值，从3减少到2
        
        # 滚动状态缓冲
        self.scroll_buffer = []
        self.scroll_buffer_size = 3  # 从5减少到3
        
        # 手指状态历史
        self.finger_history = []
        self.history_length = 5  # 从10减少到5
        
        # 节点距离阈值
        self.click_distance_threshold = 0.1  # 点击判定距离阈值
        self.scroll_trigger_threshold = 0.2  # 滚动触发阈值
        
        # 性能优化
        self.process_every_n_frames = 1  # 处理每一帧，从2改为1
        self.frame_count = 0
        
        # 添加缺失的帧缓存，用于UI显示
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # 减少帧处理的延迟
        self.last_process_time = 0
        self.min_process_interval = 0.02  # 处理间隔20ms，提高处理频率
        
        print("鼠标控制模块初始化完成")

    def get_latest_frame(self):
        """获取最新的处理后帧，供UI显示使用"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
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
        - 拇指(4)与无名指尖(16)接触
        - 向上或向下移动拇指可以进行滚动
        """
        # 获取关键点
        thumb_tip = landmarks.landmark[4]  # 拇指尖
        ring_finger_tip = landmarks.landmark[16]  # 无名指尖
        
        # 检查拇指与无名指是否触碰
        is_touching_ring = self.is_touching(thumb_tip, ring_finger_tip)
        
        # 更新历史记录
        self.finger_history.append(is_touching_ring)
        if len(self.finger_history) > self.history_length:
            self.finger_history.pop(0)
            
        # 判断是否稳定地进入滚动模式（至少2帧满足条件）
        stable_scroll = False
        if len(self.finger_history) >= 3:
            # 检查最近3帧是否有2帧满足滚动条件
            stable_scroll = sum(self.finger_history[-3:]) >= 2
            
        return stable_scroll or is_touching_ring

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
        
        # 计算安全区域的宽度和高度
        safe_width = safe_x_max - safe_x_min
        safe_height = safe_y_max - safe_y_min
        
        # 检查坐标是否在安全区域内
        if x < safe_x_min:
            # 左边缘区域映射
            screen_x = 0
        elif x > safe_x_max:
            # 右边缘区域映射
            screen_x = self.screen_width
        else:
            # 安全区域内映射 - 线性映射到整个屏幕
            normalized_x = (x - safe_x_min) / safe_width
            screen_x = int(normalized_x * self.screen_width)
        
        if y < safe_y_min:
            # 上边缘区域映射
            screen_y = 0
        elif y > safe_y_max:
            # 下边缘区域映射
            screen_y = self.screen_height
        else:
            # 安全区域内映射 - 线性映射到整个屏幕
            normalized_y = (y - safe_y_min) / safe_height
            screen_y = int(normalized_y * self.screen_height)
        
        return screen_x, screen_y

    def apply_position_smoothing(self, x, y):
        """应用位置平滑滤波
        
        多级滤波:
        1. 移动平均滤波 - 使用过去多帧的平均位置
        2. 静止检测 - 当移动幅度很小时锁定位置
        3. 低通滤波 - 对移动进行加权处理
        
        Args:
            x, y: 原始位置坐标
            
        Returns:
            平滑后的位置坐标
        """
        current_pos = (x, y)
        
        # 添加到位置缓冲区
        self.position_buffer.append(current_pos)
        if len(self.position_buffer) > self.buffer_size:
            self.position_buffer.pop(0)
            
        # 如果缓冲区尚未填满，返回当前位置
        if len(self.position_buffer) < 3:
            return x, y
            
        # 计算移动幅度 - 使用最近几帧的平均位置变化
        recent_movements = []
        for i in range(1, min(5, len(self.position_buffer))):
            prev = self.position_buffer[-i-1]
            curr = self.position_buffer[-i]
            movement = math.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            recent_movements.append(movement)
            
        avg_movement = sum(recent_movements) / len(recent_movements) if recent_movements else 0
        
        # 静止检测 - 当移动幅度低于阈值时
        if avg_movement < self.movement_threshold:
            self.stable_count += 1
            if self.stable_count >= self.stable_threshold:
                self.is_stable = True
                # 使用较早的稳定位置，减少微弱抖动
                stable_pos = self.position_buffer[-self.stable_threshold]
                return stable_pos[0], stable_pos[1]
        else:
            self.stable_count = 0
            self.is_stable = False
            
        # 移动平均滤波 - 加权平均，越近的帧权重越大
        weights = [i+1 for i in range(len(self.position_buffer))]
        total_weight = sum(weights)
        
        weighted_x = sum(pos[0] * w for pos, w in zip(self.position_buffer, weights)) / total_weight
        weighted_y = sum(pos[1] * w for pos, w in zip(self.position_buffer, weights)) / total_weight
        
        # 返回平滑后的位置
        return weighted_x, weighted_y

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
        """线程主循环，进行手部检测和鼠标控制"""
        try:
            # 只有非共享模式才需要打开摄像头
            if not self.shared_camera_mode:
                print(f"打开摄像头 ID: {self.camera_id}")
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    error_msg = f"错误：无法打开摄像头 ID {self.camera_id}"
                    print(error_msg)
                    self.status_signal.emit(error_msg)
                    return
                    
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                print("使用共享摄像头模式")
            
            self.status_signal.emit("等待帧...")
            print("鼠标控制线程开始运行")
            
            # 帧计数器和上次处理时间
            frame_counter = 0
            last_process_time = time.time()
            last_ui_update_time = 0  # 控制UI更新频率
            ui_update_interval = 0.04  # 25fps UI更新
            
            while self.running:
                # 获取帧 - 在共享模式下只使用共享帧
                frame = None
                
                if self.shared_camera_mode:
                    # 共享模式 - 从主程序获取帧
                    with self.shared_frame_lock:
                        if self.shared_frame_available and self.shared_frame is not None:
                            frame = self.shared_frame.copy()
                            self.shared_frame_available = False  # 标记已使用
                else:
                    # 非共享模式 - 从摄像头获取帧
                    if self.cap and self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if not ret:
                            error_msg = "无法读取摄像头画面"
                            print(error_msg)
                            self.status_signal.emit(error_msg)
                            break
                
                # 如果没有可用帧，等待下一次循环
                if frame is None:
                    time.sleep(0.005)  # 缩短等待时间，从0.01减少到0.005
                    continue
                
                # 性能控制 - 限制处理帧率，减少UI频闪
                current_time = time.time()
                elapsed = current_time - last_process_time
                
                # 只有在经过足够时间后才处理帧
                if elapsed < self.min_process_interval:
                    continue
                
                last_process_time = current_time
                
                # 为了提高性能，可以跳过一些帧
                self.frame_count += 1
                if self.frame_count % self.process_every_n_frames != 0:
                    continue
                
                # 翻转帧以便镜像显示（如果尚未翻转）
                if not self.shared_camera_mode:  # 共享模式下假设帧已经翻转过
                    frame = cv2.flip(frame, 1)
                
                # 转换为RGB进行处理
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # 在帧上叠加基本的状态信息
                height, width, _ = frame.shape
                status_text = "就绪"
                color = (0, 255, 0)
                
                # 准备存储关键点信息用于后续PIL绘制
                control_points = []
                mouse_coords = None
                
                # 如果检测到手部，处理鼠标控制
                if results.multi_hand_landmarks:
                    # 获取第一只手
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # 绘制手部关键点
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # 获取关键点
                    thumb_tip = hand_landmarks.landmark[4]  # 拇指尖
                    index_finger_tip = hand_landmarks.landmark[8]  # 食指尖
                    middle_finger_tip = hand_landmarks.landmark[12]  # 中指尖
                    index_finger_base = hand_landmarks.landmark[5]  # 食指根部
                    wrist = hand_landmarks.landmark[0]  # 手腕
                    
                    # 使用食指尖(8)作为鼠标控制点，替换原来的拇指尖(4)
                    hand_center = index_finger_tip
                    
                    # 将手的位置映射到屏幕坐标，使用安全区域映射
                    mouse_x, mouse_y = self.map_to_screen(hand_center.x, hand_center.y, width, height)
                    
                    # 应用自定义的多级平滑滤波
                    mouse_x, mouse_y = self.apply_position_smoothing(mouse_x, mouse_y)
                    
                    # 平滑鼠标移动 - 传统的单帧平滑
                    if self.prev_hand_center is not None:
                        mouse_x = int(self.prev_hand_center[0] + (mouse_x - self.prev_hand_center[0]) / self.smoothing)
                        mouse_y = int(self.prev_hand_center[1] + (mouse_y - self.prev_hand_center[1]) / self.smoothing)
                        
                    self.prev_hand_center = (mouse_x, mouse_y)
                    
                    # 移动鼠标（考虑避免超出屏幕边界）
                    mouse_x = max(0, min(mouse_x, self.screen_width - 1))
                    mouse_y = max(0, min(mouse_y, self.screen_height - 1))
                    
                    # 如果鼠标位置被锁定，使用锁定位置
                    if self.click_position_lock and self.locked_mouse_position:
                        mouse_x, mouse_y = self.locked_mouse_position
                    
                    # 使用直接方式移动鼠标，避免异步导致的错误
                    if not (self.click_position_lock and self.is_left_clicked and self.click_lock_frames <= self.max_click_lock_frames):
                        try:
                            # 直接移动鼠标，避免使用线程池
                            self.safe_mouse_move(mouse_x, mouse_y)
                        except Exception as e:
                            print(f"鼠标移动错误: {str(e)}")
                    
                    # 保存当前鼠标坐标信息用于后续PIL绘制
                    mouse_coords = (mouse_x, mouse_y)
                    
                    # 检查滚动手势
                    is_scroll_gesture = self.check_scroll_gesture(hand_landmarks)
                    
                    # 处理左键点击 - 拇指(4)和食指(8)触碰
                    if self.is_touching(thumb_tip, index_finger_tip) and not is_scroll_gesture:
                        if not self.is_left_clicked:
                            # 第一次检测到点击，锁定当前鼠标位置
                            if not self.click_position_lock:
                                self.locked_mouse_position = (mouse_x, mouse_y)
                                self.click_position_lock = True
                                self.click_lock_frames = 0
                            
                            # 使用锁定的位置执行点击
                            if self.locked_mouse_position:
                                self.safe_mouse_move(self.locked_mouse_position[0], self.locked_mouse_position[1])
                                pyautogui.mouseDown(button='left')
                                self.is_left_clicked = True
                                status_text = "左键点击"
                        else:
                            # 点击持续中，保持使用锁定位置
                            if self.locked_mouse_position:
                                # 点击保持一段时间后允许拖动
                                self.click_lock_frames += 1
                                if self.click_lock_frames > self.max_click_lock_frames:
                                    # 超过最大锁定帧数，允许拖动
                                    self.click_position_lock = False
                                    status_text = "拖动模式"
                                else:
                                    # 仍在锁定状态，保持鼠标位置
                                    self.safe_mouse_move(self.locked_mouse_position[0], self.locked_mouse_position[1])
                    else:
                        if self.is_left_clicked:
                            # 释放点击，取消锁定
                            pyautogui.mouseUp(button='left')
                            self.is_left_clicked = False
                            self.click_position_lock = False
                            self.locked_mouse_position = None
                            self.click_lock_frames = 0
                    
                    # 处理右键点击 - 拇指(4)和中指(12)触碰
                    if self.is_touching(thumb_tip, middle_finger_tip) and not is_scroll_gesture:
                        if not self.is_right_clicked:
                            pyautogui.mouseDown(button='right')
                            self.is_right_clicked = True
                            status_text = "右键点击"
                    else:
                        if self.is_right_clicked:
                            pyautogui.mouseUp(button='right')
                            self.is_right_clicked = False
                    
                    # 处理滚轮滚动 - 拇指(4)与无名指(16)接触
                    if is_scroll_gesture:
                        status_text = "滚轮模式"
                        
                        # 获取无名指尖
                        ring_finger_tip = hand_landmarks.landmark[16]  # 无名指尖
                        
                        # 确保从触摸开始时重置垂直位置
                        if not self.scroll_mode:
                            self.prev_index_finger_y = thumb_tip.y
                            self.scroll_mode = True
                            self.scroll_buffer = []
                        
                        # 计算垂直移动
                        if self.prev_index_finger_y is not None:
                            # 计算相对于上一帧的移动距离，直接使用拇指的垂直位移
                            scroll_amount = (thumb_tip.y - self.prev_index_finger_y) * 250  # 增大倍数，提高灵敏度
                            
                            # 添加到缓冲区
                            self.scroll_buffer.append(scroll_amount)
                            if len(self.scroll_buffer) > self.scroll_buffer_size:
                                self.scroll_buffer.pop(0)
                                
                            # 平均滚动量，减少抖动
                            if len(self.scroll_buffer) > 0:
                                avg_scroll = sum(self.scroll_buffer) / len(self.scroll_buffer)
                                if abs(avg_scroll) > 0.04:  # 减小死区，提高灵敏度(从0.05改为0.04)
                                    # 向下移动拇指，屏幕向下滚动(正值)
                                    scroll_clicks = int(avg_scroll * 12)  # 增大倍数，提高灵敏度(从10增加到12)
                                    if scroll_clicks != 0:
                                        pyautogui.scroll(scroll_clicks)
                        
                        self.prev_index_finger_y = thumb_tip.y
                    else:
                        self.scroll_mode = False
                        self.prev_index_finger_y = None
                    
                    # 标记关键控制点，收集坐标信息
                    control_points.append(self.draw_control_point(frame, thumb_tip, (0, 0, 255), "拇指(4)"))  # 红色
                    control_points.append(self.draw_control_point(frame, index_finger_tip, (0, 255, 0), "食指(8)"))  # 绿色
                    control_points.append(self.draw_control_point(frame, middle_finger_tip, (255, 0, 0), "中指(12)"))  # 蓝色
                    control_points.append(self.draw_control_point(frame, hand_landmarks.landmark[16], (255, 255, 0), "无名指(16)"))  # 黄色
                    
                else:
                    # 如果没有检测到手，重置所有鼠标状态
                    if self.is_left_clicked:
                        pyautogui.mouseUp(button='left')
                        self.is_left_clicked = False
                        
                    if self.is_right_clicked:
                        pyautogui.mouseUp(button='right')
                        self.is_right_clicked = False
                        
                    self.scroll_mode = False
                    status_text = "未检测到手"
                    color = (0, 0, 255)
                    self.prev_hand_center = None
                    self.prev_index_finger_y = None
                
                # 使用PIL绘制所有文本，包括状态信息和关键点标签
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
                                    status_font = ImageFont.truetype(font_path, 24)
                                    label_font = ImageFont.truetype(font_path, 16)
                                    hint_font = ImageFont.truetype(font_path, 18)
                                    font_found = True
                                    break
                                except Exception as font_err:
                                    # 减少字体错误日志输出
                                    continue
                                
                        if not font_found:
                            status_font = ImageFont.load_default()
                            label_font = ImageFont.load_default()
                            hint_font = ImageFont.load_default()
                    except Exception as e:
                        status_font = ImageFont.load_default()
                        label_font = ImageFont.load_default()
                        hint_font = ImageFont.load_default()
                    
                    # 添加状态文本
                    draw.text((width - 200, height - 30), status_text, font=status_font, fill=(color[2], color[1], color[0]))
                    
                    # 显示稳定状态
                    if self.is_stable:
                        stability_text = "稳定模式"
                        draw.text((width - 200, height - 60), stability_text, font=status_font, fill=(0, 255, 0))
                    
                    # 添加控制提示
                    draw.text((10, height - 80), "拇指+食指=左键", font=hint_font, fill=(255, 255, 255))
                    draw.text((10, height - 50), "拇指+中指=右键", font=hint_font, fill=(255, 255, 255))
                    draw.text((10, height - 20), "拇指+无名指=滚轮", font=hint_font, fill=(255, 255, 255))
                    
                    # 如果有坐标显示，也用PIL添加
                    if mouse_coords:
                        mouse_x, mouse_y = mouse_coords
                        draw.text((10, 30), f"X: {mouse_x}, Y: {mouse_y}", font=status_font, fill=(0, 0, 255))
                    
                    # 如果是滚动模式，显示滚动信息
                    if self.scroll_mode and len(self.scroll_buffer) > 0:
                        avg_scroll = sum(self.scroll_buffer) / len(self.scroll_buffer)
                        scroll_clicks = int(avg_scroll * 10)
                        draw.text((10, height - 120), f"滚动: {scroll_clicks}", font=status_font, fill=(255, 255, 0))
                    
                    # 绘制手指关键点标签
                    for cx, cy, label, color in control_points:
                        # OpenCV的BGR顺序转换为PIL的RGB顺序
                        pil_color = (color[2], color[1], color[0])
                        draw.text((cx + 10, cy), label, font=label_font, fill=pil_color)
                    
                    # 如果启用了安全区域显示，绘制安全区域指示框
                    if self.draw_safe_area:
                        # 计算安全区域的像素坐标
                        safe_x_min = int(width * self.safe_margin)
                        safe_x_max = int(width * (1.0 - self.safe_margin))
                        safe_y_min = int(height * self.safe_margin)
                        safe_y_max = int(height * (1.0 - self.safe_margin))
                        
                        # 绘制安全区域矩形
                        draw.rectangle(
                            [(safe_x_min, safe_y_min), (safe_x_max, safe_y_max)],
                            outline=(0, 255, 0),
                            width=2
                        )
                        
                        # 添加安全区域标签
                        draw.text(
                            (safe_x_min + 5, safe_y_min + 5),
                            "安全操作区域",
                            font=hint_font,
                            fill=(0, 255, 0)
                        )
                    
                    # 转换回OpenCV格式
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    # 如果PIL导入失败或出错，使用原始OpenCV文本渲染（不支持中文）
                    cv2.putText(frame, status_text, (width - 200, height - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                               
                    # 添加控制提示（英文）
                    cv2.putText(frame, "Thumb+Index=Left Click", (10, height - 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, "Thumb+Middle=Right Click", (10, height - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, "Thumb+Ring=Scroll", (10, height - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # 为每个关键点添加英文标签
                    for cx, cy, label, color in control_points:
                        cv2.putText(frame, label.replace("(", " ").replace(")", ""), (cx + 10, cy),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 限制UI更新频率，防止频闪
                ui_update_elapsed = current_time - last_ui_update_time
                if ui_update_elapsed >= ui_update_interval:
                    # 发送处理后的帧和状态
                    try:
                        # 先保存帧，再发送信号，防止竞态条件
                        with self.frame_lock:
                            self.latest_frame = frame.copy()
                        # 使用try-except避免信号传递失败导致的异常
                        self.frame_signal.emit(frame.copy())
                        self.status_signal.emit(status_text)
                        last_ui_update_time = current_time
                    except Exception as signal_err:
                        print(f"发送UI更新信号出错: {signal_err}")
                
                # 控制循环速度，避免CPU占用过高
                try:
                    time.sleep(0.005)  # 增加处理频率，从0.01减少到0.005
                except Exception:
                    pass  # 忽略sleep异常
                
        except Exception as e:
            error_msg = f"鼠标控制线程出错: {str(e)}"
            print(error_msg)
            self.status_signal.emit(error_msg)
            import traceback
            traceback.print_exc()
        finally:
            # 确保资源释放
            if not self.shared_camera_mode and self.cap and self.cap.isOpened():
                self.cap.release()
                print("释放摄像头资源")

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
