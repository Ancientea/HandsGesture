import sys
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time
import threading
from PyQt5.QtCore import QThread, pyqtSignal, QMutex

# 提前定义手势名称列表，确保顺序一致
GESTURE_NAMES = ["right_swipe", "left_swipe", "up_swipe", "down_swipe", "click", "pinch", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
# 选定的关键点ID - 掌根(0)、拇指(4)、食指(5,8)、中指(9,12)、无名指(13,16)、小指(17,20)
SELECTED_LANDMARKS = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]
# 更新输入维度
INPUT_DIM = len(SELECTED_LANDMARKS) * 3  # 10个关键点 × 3坐标 = 30
# 序列长度设置
SEQUENCE_LENGTH = 30  # 与train.py保持一致

class PredictionWorker:
    """预测工作器，用于执行手势预测"""
    
    def __init__(self, model):
        """初始化预测工作器
        
        Args:
            model: 加载的TensorFlow模型
        """
        self.model = model
        self.sequence = None
        self.is_running = False
        self.result_callback = None
        self.status_callback = None
    
    def set_data(self, sequence):
        """设置要预测的序列数据"""
        self.sequence = sequence
    
    def set_callbacks(self, result_callback, status_callback):
        """设置回调函数
        
        Args:
            result_callback: 预测结果回调函数
            status_callback: 状态更新回调函数
        """
        self.result_callback = result_callback
        self.status_callback = status_callback
    
    def predict(self):
        """执行预测"""
        self.is_running = True
        try:
            if self.status_callback:
                self.status_callback("开始预测...")
            print("\n=== 开始预测 ===")
            print(f"输入数据形状: {self.sequence.shape}")
            
            # 确保输入数据格式正确
            if len(self.sequence.shape) == 2:
                input_data = np.expand_dims(self.sequence, axis=0)
            else:
                input_data = self.sequence
                
            print(f"模型输入形状: {input_data.shape}")
            if self.status_callback:
                self.status_callback("模型预测中...")
            
            # 进行预测
            pred = self.model.predict(input_data, verbose=0)
            print(f"预测结果形状: {pred.shape}")
            
            # 获取所有动作的置信度
            gesture_confidences = pred[0]
            max_idx = np.argmax(gesture_confidences)
            max_confidence = gesture_confidences[max_idx]
            max_gesture = GESTURE_NAMES[max_idx]  # 使用全局定义的GESTURE_NAMES
            
            # 输出所有动作的置信度
            if self.status_callback:
                self.status_callback("处理预测结果...")
            print("\n各动作置信度:")
            for i, (gesture, conf) in enumerate(zip(GESTURE_NAMES, gesture_confidences)):
                print(f"{gesture}: {conf:.4f}")
            
            print(f"\n最大置信度动作: {max_gesture}")
            print(f"置信度: {max_confidence:.4f}")
            
            if self.is_running:
                confidence_threshold = 0.2  # 保持一致的置信度阈值
                if max_confidence > confidence_threshold:
                    if self.status_callback:
                        self.status_callback(f"识别到: {max_gesture}")
                    if self.result_callback:
                        self.result_callback(max_gesture, float(max_confidence))
                    print(f"输出动作: {max_gesture} (置信度: {max_confidence:.4f})")
                else:
                    if self.status_callback:
                        self.status_callback("置信度过低")
                    if self.result_callback:
                        self.result_callback("无动作", float(max_confidence))
                    print(f"输出: 无动作 (最大置信度 {max_confidence:.4f} 低于阈值 {confidence_threshold})")
            print("=== 预测结束 ===\n")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"预测出错: {str(e)}\n{error_details}")
            if self.status_callback:
                self.status_callback(f"预测错误: {str(e)}")
            if self.result_callback and self.is_running:
                self.result_callback("识别错误", 0.0)
        finally:
            self.is_running = False
    
    def stop(self):
        """停止预测"""
        self.is_running = False

class GestureRecognition(QThread):
    """
    手势识别线程类：负责手势模型加载与预测
    """
    status_signal = pyqtSignal(str)  # 状态信号
    gesture_signal = pyqtSignal(str, float)  # 手势结果信号
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.model = None
        self.frame_mutex = QMutex()  # 帧互斥锁，防止同时访问
        self.latest_frame = None
        
        # MediaPipe设置 - 手部检测
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        
        # 预测工作器
        self.predictor = None
        self.prediction_thread = None
        
        # 初始化模型
        self.init_model()
        
        print("手势识别模块初始化完成")
    
    def init_model(self):
        """初始化模型"""
        try:
            # 尝试加载.h5格式的模型
            model_path = "model_selected_landmarks.h5"
            if not os.path.exists(model_path):
                model_path = "best_model.h5"  # 回退到原始模型
                print(f"警告: 找不到选定关键点的模型，尝试加载原始模型 {model_path}")
            
            if not os.path.exists(model_path):
                print(f"错误：找不到模型文件 {model_path}")
                self.status_signal.emit("错误：找不到模型文件")
                self.model = None
                return
            
            print(f"正在加载模型 {model_path}...")
            print(f"TensorFlow版本: {tf.version.VERSION}")
            
            # 加载.h5格式的模型
            try:
                # 使用 tf.keras.models.load_model 加载模型
                self.model = tf.keras.models.load_model(model_path, compile=False)
                print("模型加载成功")
                print(f"使用关键点IDs: {SELECTED_LANDMARKS}")
                print(f"输入维度: {INPUT_DIM}")
                print(f"序列长度: {SEQUENCE_LENGTH}")
                
                # 初始化预测工作器
                self.predictor = PredictionWorker(self.model)
                
            except Exception as e:
                print(f"模型加载失败: {str(e)}")
                self.status_signal.emit("模型加载失败")
                self.model = None
                return
            
            # 编译模型 - 使用与 TensorFlow 2.10.0 兼容的配置
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("模型编译成功")
            print(f"模型输入形状: {self.model.input_shape}")
            print(f"模型输出形状: {self.model.output_shape}")
            
        except Exception as e:
            print(f"模型初始化失败: {str(e)}")
            self.status_signal.emit(f"模型初始化失败: {str(e)}")
    
    def process_shared_frame(self, frame):
        """处理从主程序共享的帧"""
        try:
            if frame is not None:
                self.frame_mutex.lock()
                self.latest_frame = frame.copy()  # 创建副本避免引用问题
                self.frame_mutex.unlock()
        except Exception as e:
            print(f"处理共享帧出错: {str(e)}")
    
    def get_latest_frame(self):
        """获取最新的处理后帧"""
        self.frame_mutex.lock()
        if self.latest_frame is not None:
            frame = self.latest_frame.copy()
        else:
            frame = None
        self.frame_mutex.unlock()
        return frame
    
    def start_recognition(self):
        """启动手势识别"""
        if self.model is None:
            self.status_signal.emit("模型未加载，无法启动识别")
            return False
        
        self.running = True
        if not self.isRunning():
            self.start()
        
        self.status_signal.emit("手势识别已启动")
        return True
    
    def stop_recognition(self):
        """停止手势识别"""
        self.running = False
        
        # 停止当前的预测线程
        if self.prediction_thread and self.prediction_thread.is_alive():
            self.predictor.stop()
            try:
                # 最多等待0.5秒，避免线程卡死
                self.prediction_thread.join(timeout=0.5)
            except Exception as e:
                print(f"停止预测线程出错: {str(e)}")
        
        # 等待线程结束，设置超时
        if self.isRunning():
            try:
                self.wait(1000)  # 最多等待1秒
            except Exception as e:
                print(f"等待线程结束出错: {str(e)}")
        
        self.status_signal.emit("手势识别已停止")
    
    def predict_gesture(self, sequence):
        """预测手势
        
        Args:
            sequence: 要预测的序列数据
        """
        if self.model is None:
            self.status_signal.emit("模型未加载，无法预测")
            return
        
        try:
            # 停止现有的预测线程
            if self.prediction_thread and self.prediction_thread.is_alive():
                self.predictor.stop()
                try:
                    # 最多等待0.5秒
                    self.prediction_thread.join(timeout=0.5)
                except Exception as e:
                    print(f"停止预测线程出错: {str(e)}")
            
            # 准备新的预测
            self.predictor.set_data(sequence)
            self.predictor.set_callbacks(
                result_callback=lambda gesture, conf: self.gesture_signal.emit(gesture, conf),
                status_callback=lambda status: self.status_signal.emit(status)
            )
            
            # 创建并启动新线程
            self.prediction_thread = threading.Thread(target=self.predictor.predict)
            self.prediction_thread.daemon = True
            
            # 使用异常处理启动线程
            try:
                self.prediction_thread.start()
            except Exception as e:
                print(f"启动预测线程出错: {str(e)}")
                self.status_signal.emit(f"启动预测线程出错: {str(e)}")
            
        except Exception as e:
            self.status_signal.emit(f"预测出错: {str(e)}")
            print(f"预测出错: {str(e)}")
    
    def run(self):
        """线程主运行函数"""
        try:
            print("手势识别线程已启动")
            self.status_signal.emit("线程已启动")
            
            while self.running:
                # 暂停一下避免CPU占用过高
                try:
                    time.sleep(0.01)
                except Exception:
                    pass
                
        except Exception as e:
            self.status_signal.emit(f"运行出错: {str(e)}")
            print(f"手势识别线程运行出错: {str(e)}")
        finally:
            print("手势识别线程已停止") 