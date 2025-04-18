# 手势识别系统

这是一个基于深度学习的手势识别系统，使用MediaPipe进行手部关键点检测，使用TensorFlow/Keras构建和训练深度学习模型。

## 功能特点

- 实时手势识别
- 支持6种手势：右滑、左滑、上滑、下滑、点击、捏合
- 使用BiLSTM模型进行序列分类
- 基于PyQt5的图形用户界面
- 支持摄像头实时预览

## 环境要求

- Python 3.9+
- TensorFlow 2.12.0
- Keras 2.12.0
- OpenCV
- MediaPipe
- PyQt5
- NumPy
- scikit-learn

## 安装步骤

1. 创建并激活虚拟环境（推荐）：
```bash
conda create -n hands python=3.9
conda activate hands
```

2. 安装依赖包：
```bash
pip install tensorflow==2.12.0
pip install opencv-python
pip install mediapipe
pip install PyQt5
pip install numpy
pip install scikit-learn
```

## 项目结构

```
HandsGesture/
├── README.md
├── main.py              # 主程序
├── train.py            # 训练脚本
├── data_collector.py   # 数据采集工具
├── method.py           # 工具方法
├── best_model.h5       # 训练好的模型
└── gesture_data/       # 手势数据目录
    ├── click/         # 点击手势数据
    ├── pinch/         # 捏合手势数据
    ├── left_swipe/    # 左滑手势数据
    ├── right_swipe/   # 右滑手势数据
    ├── up_swipe/      # 上滑手势数据
    └── down_swipe/    # 下滑手势数据
```

## 使用方法

1. 训练模型：
```bash
python train.py
```

2. 运行主程序：
```bash
python main.py
```

3. 在程序界面中：
   - 输入摄像头ID（默认为0）
   - 点击"开始识别"按钮
   - 在摄像头前做出手势
   - 程序会实时显示识别结果

## 手势说明

- 右滑：手掌向右滑动
- 左滑：手掌向左滑动
- 上滑：手掌向上滑动
- 下滑：手掌向下滑动
- 点击：食指点击动作
- 捏合：拇指和食指捏合动作

## 注意事项

1. 确保摄像头正常工作
2. 手势动作要清晰明确
3. 保持适当的光线条件
4. 避免快速移动造成模糊

## 模型信息

- 输入：30帧序列，每帧63个特征点（21个关键点 × 3个坐标）
- 模型结构：BiLSTM + LayerNormalization + Dropout
- 训练准确率：>99%
- 测试准确率：>96%

## 许可证

MIT License

## 作者

[您的名字]

## 联系方式

[您的联系方式] 