import sys
import time
import threading
import pyautogui
import keyboard
from PyQt5.QtCore import QObject, pyqtSignal, QThread

class HandsControlThread(QThread):
    """
    手势控制线程：负责将识别的手势转换为系统控制命令
    """
    status_signal = pyqtSignal(str)  # 状态信号
    mode_switch_signal = pyqtSignal(bool)  # 模式切换信号，True为启动鼠标模式，False为启动手势控制模式
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.last_gesture = None
        self.last_gesture_time = 0
        self.cooldown = 1.0  # 手势冷却时间，防止误触发
        self.gesture_mapping = {
            "up_swipe": self.volume_up,        # 上滑 - 音量加
            "down_swipe": self.volume_down,    # 下滑 - 音量减
            "right_swipe": self.right_arrow,   # 右滑 - 右箭头
            "left_swipe": self.left_arrow,     # 左滑 - 左箭头
            "click": self.space,               # 点击 - 空格
            "two": self.toggle_fullscreen      # 二 - 全屏/退出全屏
        }
        
        # 添加模式切换检测
        self.six_gesture_times = []  # 记录"6"手势的时间
        self.mode_switch_cooldown = 3.0  # 模式切换冷却时间
        self.last_mode_switch_time = 0  # 上次模式切换时间
        self.mouse_mode = False  # 当前是否为鼠标模式
        
        # 设置pyautogui安全措施
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
        self.status_signal.emit("手势控制模块已初始化")
        print("手势控制模块已初始化")
    
    def run(self):
        """线程主循环"""
        self.running = True
        self.status_signal.emit("手势控制已启动")
        print("手势控制已启动")
        
        # 主循环在process_gesture中处理，这里只用于保持线程活跃
        while self.running:
            time.sleep(0.1)
    
    def process_gesture(self, gesture, confidence):
        """处理接收到的手势"""
        if not self.running:
            return
            
        current_time = time.time()
        
        # 处理"6"手势的模式切换检测 - 调低置信度要求，确保更容易检测
        if gesture == "six" and confidence > 0.2:  # 从0.25降低到0.2
            print(f"检测到'6'手势，置信度: {confidence:.2f}")
            self.six_gesture_times.append(current_time)
            
            # 添加更详细的调试信息
            print(f"'6'手势历史: {[f'{t-self.six_gesture_times[0]:.1f}s' for t in self.six_gesture_times]}")
            
            # 只保留最近5秒内的手势记录
            self.six_gesture_times = [t for t in self.six_gesture_times if current_time - t <= 5.0]
            
            # 检查是否有连续两个"6"手势
            if len(self.six_gesture_times) >= 2:
                # 确保两个手势的时间间隔合理(0.5-5秒)，避免误触发
                time_diff = self.six_gesture_times[-1] - self.six_gesture_times[-2]
                print(f"两个'6'手势间隔: {time_diff:.2f}秒")
                
                if 0.5 <= time_diff <= 5.0:
                    # 确保模式切换有冷却时间
                    if current_time - self.last_mode_switch_time > self.mode_switch_cooldown:
                        self.last_mode_switch_time = current_time
                        self.mouse_mode = not self.mouse_mode
                        
                        if self.mouse_mode:
                            self.status_signal.emit("切换到鼠标控制模式")
                            print("切换到鼠标控制模式")
                        else:
                            self.status_signal.emit("切换到手势控制模式")
                            print("切换到手势控制模式")
                        
                        # 发出模式切换信号
                        self.mode_switch_signal.emit(self.mouse_mode)
                        
                        # 清空手势记录
                        self.six_gesture_times = []
                        print("已清空手势记录，等待新的手势输入")
                        return
                    else:
                        print(f"模式切换冷却中，距离上次切换: {current_time - self.last_mode_switch_time:.1f}秒")
                else:
                    print(f"手势间隔 {time_diff:.2f}秒 不在有效范围(0.5-5.0秒)内")
        
        # 如果处于鼠标模式，则不处理普通手势
        if self.mouse_mode:
            return
            
        # 只处理我们映射的手势
        if not gesture or gesture not in self.gesture_mapping:
            return
        
        # 手势置信度阈值检查
        if confidence < 0.35:  # 可以根据实际情况调整阈值
            return
        
        # 检查冷却时间
        if (current_time - self.last_gesture_time < self.cooldown):
            return
        
        # 记录本次手势时间
        self.last_gesture_time = current_time
        self.last_gesture = gesture
        
        # 执行映射的动作
        try:
            action_func = self.gesture_mapping[gesture]
            # 创建一个新线程执行动作，避免阻塞主线程
            threading.Thread(target=action_func).start()
            self.status_signal.emit(f"执行动作: {gesture}")
            print(f"执行动作: {gesture}")
        except Exception as e:
            self.status_signal.emit(f"动作执行错误: {str(e)}")
            print(f"动作执行错误: {str(e)}")
    
    def volume_up(self):
        """增加音量"""
        try:
            keyboard.press_and_release('volume up')
            self.status_signal.emit("音量增加")
        except Exception as e:
            self.status_signal.emit(f"音量增加失败: {str(e)}")
    
    def volume_down(self):
        """减小音量"""
        try:
            keyboard.press_and_release('volume down')
            self.status_signal.emit("音量减小")
        except Exception as e:
            self.status_signal.emit(f"音量减小失败: {str(e)}")
    
    def right_arrow(self):
        """按下右箭头"""
        try:
            keyboard.press_and_release('right')
            self.status_signal.emit("右箭头")
        except Exception as e:
            self.status_signal.emit(f"右箭头失败: {str(e)}")
    
    def left_arrow(self):
        """按下左箭头"""
        try:
            keyboard.press_and_release('left')
            self.status_signal.emit("左箭头")
        except Exception as e:
            self.status_signal.emit(f"左箭头失败: {str(e)}")
    
    def space(self):
        """按下空格键"""
        try:
            keyboard.press_and_release('space')
            self.status_signal.emit("空格")
        except Exception as e:
            self.status_signal.emit(f"空格失败: {str(e)}")
    
    def toggle_fullscreen(self):
        """切换全屏状态"""
        try:
            keyboard.press_and_release('f11')  # 大多数媒体播放器使用F11切换全屏
            self.status_signal.emit("切换全屏")
        except Exception as e:
            self.status_signal.emit(f"切换全屏失败: {str(e)}")
    
    def start_control(self):
        """启动手势控制"""
        if not self.isRunning():
            self.start()
        else:
            self.running = True
            self.status_signal.emit("手势控制已恢复")
    
    def stop_control(self):
        """停止手势控制"""
        self.running = False
        self.status_signal.emit("手势控制已停止")
        
        # 等待线程结束
        if self.isRunning():
            self.wait(500)  # 等待最多500ms

# 测试代码
if __name__ == "__main__":
    print("手势控制模块测试")
    control = HandsControlThread()
    control.start_control()
    
    # 模拟手势输入进行测试
    test_gestures = ["up_swipe", "down_swipe", "right_swipe", "left_swipe", "click", "two"]
    
    for gesture in test_gestures:
        print(f"测试手势: {gesture}")
        control.process_gesture(gesture, 0.8)
        time.sleep(2)  # 等待动作执行
    
    control.stop_control()
    print("测试完成")
