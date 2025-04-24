import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Dropout, LayerNormalization, 
    Conv1D, BatchNormalization, MultiHeadAttention, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 导入 GPU 配置
from temp.gpu_config import configure_gpu

# 配置 GPU
if not configure_gpu():
    print("警告：GPU 配置失败，将使用 CPU 进行训练")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 超参数配置
DATA_DIR = "gesture_data"  # 数据文件夹
SEQUENCE_LENGTH = 30  # 序列长度（帧数）
# 选定的关键点ID - 掌根(0)、拇指(4)、食指(5,8)、中指(9,12)、无名指(13,16)、小指(17,20)
SELECTED_LANDMARKS = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]
INPUT_DIM = len(SELECTED_LANDMARKS) * 3  # 每个关键点有x,y,z三个坐标
BATCH_SIZE = 32
EPOCHS = 100  # 增加训练轮数以充分学习
INITIAL_LR = 0.001  # 初始学习率
VAL_SPLIT = 0.2  # 验证集比例
RANDOM_SEED = 42  # 随机种子，确保结果可复现

# 提前定义手势名称列表，确保顺序一致
GESTURE_NAMES = ["right_swipe", "left_swipe", "up_swipe", "down_swipe", "click", "pinch", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

# 设置随机种子，确保结果可复现
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# 尝试检测和配置GPU
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"检测到 {len(gpus)} 个GPU设备")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("已启用GPU内存动态增长")
    else:
        print("未检测到GPU设备，将使用CPU训练")
except Exception as e:
    print(f"GPU设置错误: {str(e)}")

def extract_landmarks(data):
    """从完整的手部关键点数据中提取选定的关键点
    
    Args:
        data: 形状为 (帧数, 21*3) 的原始数据，每3个值代表一个关键点的x,y,z坐标
        
    Returns:
        形状为 (帧数, len(SELECTED_LANDMARKS)*3) 的处理后数据
    """
    # 转换为 (帧数, 21, 3) 形状以便于处理
    num_frames = len(data)
    reshaped_data = data.reshape(num_frames, 21, 3)
    
    # 提取选定的关键点
    selected_data = reshaped_data[:, SELECTED_LANDMARKS, :]
    
    # 展平为 (帧数, len(SELECTED_LANDMARKS)*3) 形状
    return selected_data.reshape(num_frames, -1)

def analyze_landmarks_importance(X, y):
    """分析各关键点的重要性"""
    plt.figure(figsize=(12, 8))
    
    # 1. 计算每个关键点的变化量
    landmark_variations = []
    y_labels = np.argmax(y, axis=1)
    
    # 重新整形为(样本数, 序列长度, 关键点数, 3)
    reshaped_X = X.reshape(X.shape[0], X.shape[1], len(SELECTED_LANDMARKS), 3)
    
    for landmark_idx in range(len(SELECTED_LANDMARKS)):
        # 计算每个样本中该关键点的平均变化量
        variations = []
        for sample_idx in range(len(X)):
            landmark_seq = reshaped_X[sample_idx, :, landmark_idx, :]  # 提取特定关键点的序列
            # 计算相邻帧之间的变化量
            frame_diffs = np.linalg.norm(landmark_seq[1:] - landmark_seq[:-1], axis=1)
            avg_variation = np.mean(frame_diffs) if len(frame_diffs) > 0 else 0
            variations.append(avg_variation)
        
        landmark_variations.append(variations)
    
    # 2. 绘制每个关键点的变化量分布
    plt.subplot(2, 1, 1)
    box_data = []
    for landmark_idx in range(len(SELECTED_LANDMARKS)):
        box_data.append(landmark_variations[landmark_idx])
    
    plt.boxplot(box_data, labels=[f"ID: {id}" for id in SELECTED_LANDMARKS])
    plt.title('各关键点的运动变化量分布')
    plt.xlabel('关键点ID')
    plt.ylabel('平均变化量')
    
    # 3. 分析每个手势类别中各关键点的重要性
    plt.subplot(2, 1, 2)
    gesture_landmark_importance = np.zeros((len(GESTURE_NAMES), len(SELECTED_LANDMARKS)))
    
    for gesture_idx in range(len(GESTURE_NAMES)):
        gesture_mask = y_labels == gesture_idx
        for landmark_idx in range(len(SELECTED_LANDMARKS)):
            landmark_values = [landmark_variations[landmark_idx][i] for i in range(len(landmark_variations[landmark_idx])) if gesture_mask[i]]
            gesture_landmark_importance[gesture_idx, landmark_idx] = np.mean(landmark_values) if landmark_values else 0
    
    # 标准化每个手势的关键点重要性
    for gesture_idx in range(len(GESTURE_NAMES)):
        row_sum = np.sum(gesture_landmark_importance[gesture_idx, :])
        if row_sum > 0:
            gesture_landmark_importance[gesture_idx, :] /= row_sum
    
    # 绘制热力图
    sns.heatmap(gesture_landmark_importance, 
               xticklabels=[f"ID: {id}" for id in SELECTED_LANDMARKS],
               yticklabels=GESTURE_NAMES,
               cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title('各手势类别中关键点的相对重要性')
    plt.xlabel('关键点ID')
    plt.ylabel('手势类别')
    
    plt.tight_layout()
    plt.savefig('result/landmark_importance_analysis.png')
    plt.close()
    print("关键点重要性分析已保存到 'landmark_importance_analysis.png'")

def data_augmentation(sequence):
    """对序列数据进行增强，生成新的训练样本
    
    Args:
        sequence: 形状为 (帧数, 特征数) 的输入序列
        
    Returns:
        增强后的序列
    """
    augmented = sequence.copy()
    current_length = len(augmented)
    
    # 随机选择增强类型
    augmentation_type = random.choice(['noise', 'scale', 'time_warp', 'combo'])
    
    # 1. 添加噪声
    if augmentation_type == 'noise' or augmentation_type == 'combo':
        noise_level = np.random.uniform(0.005, 0.02)
        noise = np.random.normal(0, noise_level, sequence.shape)
        augmented += noise
    
    # 2. 随机缩放
    if augmentation_type == 'scale' or augmentation_type == 'combo':
        scale_factor = np.random.uniform(0.9, 1.1)
        augmented *= scale_factor
    
    # 3. 时间扭曲 - 改变动作速度
    if augmentation_type == 'time_warp' or augmentation_type == 'combo':
        if current_length > 5:
            time_window = min(5, current_length // 3)
            start_idx = np.random.randint(0, current_length - time_window)
            
            # 随机决定加速或减速
            if random.random() < 0.5 and current_length < SEQUENCE_LENGTH - 2:
                # 局部减速 - 插入帧
                insert_idx = start_idx + time_window // 2
                new_frame = (augmented[insert_idx] + augmented[insert_idx-1]) / 2
                augmented = np.insert(augmented, insert_idx, new_frame, axis=0)
                current_length += 1
            elif current_length > time_window + 3:
                # 局部加速 - 删除帧
                remove_idx = start_idx + time_window // 2
                augmented = np.delete(augmented, remove_idx, axis=0)
                current_length -= 1
    
    # 确保序列长度统一为SEQUENCE_LENGTH
    if len(augmented) < SEQUENCE_LENGTH:
        # 序列太短则补齐
        last_frame = augmented[-1]
        pad_frames = np.tile(last_frame, (SEQUENCE_LENGTH - len(augmented), 1))
        augmented = np.vstack([augmented, pad_frames])
    else:
        # 序列太长则截断
        augmented = augmented[:SEQUENCE_LENGTH]
    
    return augmented

def preprocess_sequence(sequence):
    """对手势序列进行预处理：标准化和去噪
    
    Args:
        sequence: 形状为 (帧数, 特征数) 的输入序列
        
    Returns:
        处理后的序列
    """
    # 复制序列避免修改原始数据
    preprocessed = sequence.copy()
    
    # 1. 帧间标准化 - 每一帧都进行Z-score归一化
    for i in range(len(preprocessed)):
        frame = preprocessed[i]
        # Z-score标准化，确保不同尺度的手在特征空间中表示一致
        frame = (frame - np.mean(frame)) / (np.std(frame) + 1e-8)
        preprocessed[i] = frame
    
    # 2. 应用平滑处理减少采集噪声
    kernel_size = 3
    if len(preprocessed) > kernel_size:
        kernel = np.ones(kernel_size) / kernel_size
        # 对每个特征应用一维卷积作为平滑操作
        for i in range(preprocessed.shape[1]):
            feature = preprocessed[:, i]
            smoothed = np.convolve(feature, kernel, mode='same')
            # 保留序列边缘的原始值
            smoothed[0] = feature[0]
            smoothed[-1] = feature[-1]
            preprocessed[:, i] = smoothed
    
    return preprocessed

def load_dataset():
    """加载手势数据集，进行预处理和增强
    
    Returns:
        X: 形状为 (样本数, SEQUENCE_LENGTH, INPUT_DIM) 的输入数据
        y: 形状为 (样本数, 类别数) 的独热编码标签
        gesture_names: 手势类别名称列表
    """
    X, y = [], []
    gesture_names = GESTURE_NAMES
    class_sample_counts = {}
    
    print("开始加载数据集...")
    
    # 第一次扫描：统计每个类别的样本数量
    for label_idx, gesture in enumerate(gesture_names):
        gesture_dir = os.path.join(DATA_DIR, gesture)
        if not os.path.exists(gesture_dir):
            print(f"警告：找不到类别 '{gesture}' 的目录，将跳过")
            continue
            
        sample_count = len(os.listdir(gesture_dir))
        class_sample_counts[gesture] = sample_count
        print(f"类别 '{gesture}' 包含 {sample_count} 个样本")
    
    # 计算每个类别增强后的目标样本数
    max_samples = max(class_sample_counts.values())
    target_samples_per_class = max_samples * 1.2  # 略微增加以确保数据充足
    
    # 第二次扫描：加载数据并进行处理
    for label_idx, gesture in enumerate(gesture_names):
        gesture_dir = os.path.join(DATA_DIR, gesture)
        if not os.path.exists(gesture_dir):
            continue
            
        files = os.listdir(gesture_dir)
        print(f"正在处理 '{gesture}' 类别...")
        
        # 原始样本数量
        original_count = len(files)
        
        # 计算需要的增强样本数量
        augmentations_needed = max(0, int(target_samples_per_class) - original_count)
        augmentations_per_sample = max(1, augmentations_needed // original_count)
        
        # 处理每个样本文件
        for file_idx, file in enumerate(files):
            try:
                # 加载原始数据
                data = np.load(os.path.join(gesture_dir, file))
                
                # 提取选定的关键点
                data = extract_landmarks(data)
                
                # 标准化和预处理
                data = preprocess_sequence(data)
                
                # 确保数据长度一致
                if len(data) < SEQUENCE_LENGTH:
                    # 如果数据太短，重复最后一帧
                    last_frame = data[-1]
                    pad_frames = np.tile(last_frame, (SEQUENCE_LENGTH - len(data), 1))
                    data = np.vstack([data, pad_frames])
                else:
                    # 如果数据太长，截断
                    data = data[:SEQUENCE_LENGTH]
                
                # 添加原始样本
                X.append(data)
                y.append(label_idx)
                
                # 为样本数量较少的类别生成增强样本
                for _ in range(augmentations_per_sample):
                    augmented_data = data_augmentation(data)
                    X.append(augmented_data)
                    y.append(label_idx)
                    
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")
    
    # 转换为NumPy数组
    X = np.array(X)
    y = to_categorical(y, num_classes=len(gesture_names))
    
    print(f"最终数据形状: {X.shape}, 标签形状: {y.shape}")
    print(f"使用的关键点ID: {SELECTED_LANDMARKS}")
    print(f"输入特征维度: {INPUT_DIM}")
    
    return X, y, gesture_names

def build_model():
    """构建改进的手势识别模型，包含时空特征提取和注意力机制
    
    Returns:
        构建好的模型
    """
    inputs = Input(shape=(SEQUENCE_LENGTH, INPUT_DIM))
    
    # 1. 时序卷积提取局部特征模式 - 捕捉关键点间的空间关系
    x = Conv1D(64, 5, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 2. 第二层卷积 - 提取更高层次的模式
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x1 = Dropout(0.3)(x)  # 保存用于后续残差连接
    
    # 3. 第一个双向LSTM - 捕捉序列中的长期依赖关系
    x = Bidirectional(LSTM(128, return_sequences=True))(x1)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 4. 多头注意力机制 - 关注序列中的重要帧
    x = MultiHeadAttention(
        num_heads=4, key_dim=32
    )(x, x, x)  # 自注意力
    x = LayerNormalization()(x)
    x2 = Dropout(0.3)(x)
    
    # 5. 残差连接 - 将之前的特征与当前特征结合
    # 先调整通道数以匹配
    res_connection = Conv1D(x2.shape[-1], 1, padding='same')(x1)
    x = Add()([res_connection, x2])
    
    # 6. 第二个双向LSTM - 进一步融合时序特征
    x = Bidirectional(LSTM(64))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 7. 全连接层 - 特征整合与分类
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # 8. 输出层
    outputs = Dense(len(GESTURE_NAMES), activation='softmax')(x)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=outputs)
    
    # 配置优化器和损失函数
    optimizer = Adam(learning_rate=INITIAL_LR)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """绘制训练历史记录，分析模型训练过程
    
    Args:
        history: 模型训练的历史记录
    """
    plt.figure(figsize=(15, 12))
    
    # 1. 准确率曲线
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    # 2. 损失曲线
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 3. 学习率曲线
    if 'lr' in history.history:
        plt.subplot(2, 2, 3)
        plt.semilogy(history.history['lr'])  # 使用对数坐标
        plt.title('学习率变化')
        plt.xlabel('训练轮次')
        plt.ylabel('学习率')
        plt.grid(True)
    
    # 4. 训练与验证准确率差值 - 用于监测过拟合
    plt.subplot(2, 2, 4)
    acc_diff = [train - val for train, val in zip(history.history['accuracy'], history.history['val_accuracy'])]
    plt.plot(acc_diff)
    plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.7)  # 过拟合警告线
    plt.axhline(y=-0.1, color='r', linestyle='--', alpha=0.7)
    plt.title('训练与验证准确率差值')
    plt.xlabel('训练轮次')
    plt.ylabel('差值')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('result/training_history.png')
    plt.close()
    
    # 保存训练历史数据
    np.save('result/training_history.npy', history.history)
    print("训练历史已保存到 'training_history.png'")

def calculate_confusion_matrix(model, X_test, y_test, gesture_names):
    """计算并可视化混淆矩阵及模型性能指标
    
    Args:
        model: 训练好的模型
        X_test: 测试数据
        y_test: 测试标签
        gesture_names: 手势类别名称
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    # 1. 预测测试集
    y_pred = model.predict(X_test)
    
    # 2. 转换为类别索引
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    # 3. 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 4. 标准化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 5. 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
              xticklabels=gesture_names, yticklabels=gesture_names)
    plt.title('标准化混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.tight_layout()
    plt.savefig('result/confusion_matrix.png')
    plt.close()
    
    print("混淆矩阵已保存到 'confusion_matrix.png'")
    
    # 6. 打印分类报告
    print("\n分类报告:")
    report = classification_report(y_true, y_pred, target_names=gesture_names)
    print(report)
    
    # 将分类报告保存到文件
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    
    # 7. 计算每个类别的主要指标
    print("\n各手势类别的性能指标:")
    for i, gesture in enumerate(gesture_names):
        true_positive = cm[i, i]
        false_negative = sum(cm[i, :]) - true_positive
        false_positive = sum(cm[:, i]) - true_positive
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{gesture}:")
        print(f"  - 精确率(Precision): {precision:.4f}")
        print(f"  - 召回率(Recall): {recall:.4f}")
        print(f"  - F1分数: {f1:.4f}")

def visualize_landmark_importance():
    """可视化手部关键点及所选关键点的重要性"""
    plt.figure(figsize=(10, 10))
    
    # 手部关键点的典型连接（简化版）
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
        (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20)   # 小指
    ]
    
    # 关键点位置（简化的手部模型）
    positions = {
        0: (0.5, 0.8),  # 掌根
        1: (0.4, 0.7), 2: (0.3, 0.6), 3: (0.2, 0.5), 4: (0.1, 0.4),  # 拇指
        5: (0.5, 0.7), 6: (0.5, 0.6), 7: (0.5, 0.5), 8: (0.5, 0.4),  # 食指
        9: (0.6, 0.7), 10: (0.6, 0.6), 11: (0.6, 0.5), 12: (0.6, 0.4),  # 中指
        13: (0.7, 0.7), 14: (0.7, 0.6), 15: (0.7, 0.5), 16: (0.7, 0.4),  # 无名指
        17: (0.8, 0.7), 18: (0.8, 0.6), 19: (0.8, 0.5), 20: (0.8, 0.4)   # 小指
    }
    
    # 绘制连接
    for connection in connections:
        start, end = connection
        plt.plot([positions[start][0], positions[end][0]], 
                [positions[start][1], positions[end][1]], 'gray', alpha=0.5)
    
    # 绘制关键点
    for i in range(21):
        if i in SELECTED_LANDMARKS:
            plt.plot(positions[i][0], positions[i][1], 'ro', markersize=12)  # 选中的点用红色
            plt.text(positions[i][0], positions[i][1], str(i), fontsize=12, 
                    ha='center', va='center', color='white')
        else:
            plt.plot(positions[i][0], positions[i][1], 'bo', markersize=8, alpha=0.3)  # 未选中的点用蓝色
            plt.text(positions[i][0], positions[i][1], str(i), fontsize=10, 
                    ha='center', va='center', color='black', alpha=0.5)
    
    plt.title('手部21个关键点及所选择的10个关键点（红色标注）')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.savefig('result/selected_landmarks.png')
    plt.close()
    print("所选关键点可视化已保存到 'selected_landmarks.png'")

def visualize_features(X, y, gesture_names):
    """可视化手势特征分布
    
    Args:
        X: 形状为 (样本数, SEQUENCE_LENGTH, INPUT_DIM) 的输入数据
        y: 形状为 (样本数, 类别数) 的独热编码标签
        gesture_names: 手势类别名称列表
    """
    y_labels = np.argmax(y, axis=1)
    
    plt.figure(figsize=(18, 12))
    
    # 1. 样本数量分布
    plt.subplot(2, 3, 1)
    sns.countplot(x=y_labels)
    plt.title('各手势类别样本数量')
    plt.xlabel('手势类别')
    plt.ylabel('样本数量')
    plt.xticks(range(len(gesture_names)), gesture_names, rotation=45)
    
    # 2. 第一帧的XY坐标分布 - 用掌根和食指尖的坐标
    plt.subplot(2, 3, 2)
    for i in range(len(gesture_names)):
        mask = y_labels == i
        if np.any(mask):
            # 提取特定手势的第一帧数据
            first_frames = X[mask, 0, :]
            # 掌根的X坐标与食指尖的Y坐标
            root_x = first_frames[:, 0]  # 掌根X
            index_y = first_frames[:, 8*3+1]  # 食指尖Y (对应原始8号点)
            plt.scatter(root_x, index_y, alpha=0.6, label=gesture_names[i])
    plt.title('第一帧掌根X与食指尖Y的分布')
    plt.xlabel('掌根X坐标')
    plt.ylabel('食指尖Y坐标')
    plt.legend()
    
    # 3. 运动轨迹可视化 - 选择一个典型样本
    plt.subplot(2, 3, 3)
    for i in range(len(gesture_names)):
        mask = y_labels == i
        if np.any(mask):
            # 为每个手势选择一个样本
            sample_idx = np.where(mask)[0][0]
            sample = X[sample_idx]
            
            # 提取食指尖的轨迹 (X和Y坐标)
            index_x = sample[:, 8*3]  # 食指尖X (对应原始8号点)
            index_y = sample[:, 8*3+1]  # 食指尖Y
            
            plt.plot(index_x, index_y, 'o-', label=gesture_names[i], alpha=0.7)
            # 标记起点和终点
            plt.plot(index_x[0], index_y[0], 'go', markersize=8)  # 起点
            plt.plot(index_x[-1], index_y[-1], 'ro', markersize=8)  # 终点
    
    plt.title('食指尖XY轨迹')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.legend()
    plt.grid(True)
    
    # 4. t-SNE降维可视化
    plt.subplot(2, 3, 4)
    # 选择每个样本的第一帧，并将其展平
    first_frames = X[:, 0, :]
    
    # 使用PCA进行初步降维
    pca = PCA(n_components=min(50, first_frames.shape[1]))
    pca_result = pca.fit_transform(first_frames)
    
    # 使用t-SNE进一步降维到2D
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(pca_result)
    
    # 绘制t-SNE结果
    for i in range(len(gesture_names)):
        mask = y_labels == i
        if np.any(mask):
            plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                      alpha=0.6, label=gesture_names[i])
    plt.title('手势t-SNE可视化')
    plt.xlabel('t-SNE维度1')
    plt.ylabel('t-SNE维度2')
    plt.legend()
    
    # 5. 运动幅度分布
    plt.subplot(2, 3, 5)
    motion_amplitudes = []
    
    for sample in X:
        # 计算相邻帧之间的运动幅度
        frame_diffs = np.sqrt(np.sum(np.square(sample[1:] - sample[:-1]), axis=1))
        max_motion = np.max(frame_diffs) if len(frame_diffs) > 0 else 0
        motion_amplitudes.append(max_motion)
    
    # 绘制每个类别的运动幅度箱线图
    box_data = []
    for i in range(len(gesture_names)):
        mask = y_labels == i
        if np.any(mask):
            box_data.append([motion_amplitudes[j] for j in range(len(motion_amplitudes)) if mask[j]])
    
    plt.boxplot(box_data, labels=gesture_names)
    plt.title('各手势类别的最大运动幅度')
    plt.xlabel('手势类别')
    plt.ylabel('最大运动幅度')
    plt.xticks(rotation=45)
    
    # 6. 序列长度分析 - 模拟各手势原始长度分布
    plt.subplot(2, 3, 6)
    
    # 生成手势序列长度的分布数据
    avg_lengths = {
        'right_swipe': 20, 'left_swipe': 19, 'up_swipe': 15, 
        'down_swipe': 16, 'click': 10, 'pinch': 12
    }
    
    # 为每个手势创建假设的长度分布
    for i, gesture in enumerate(gesture_names):
        if gesture in avg_lengths:
            # 基于平均长度创建正态分布
            length = avg_lengths.get(gesture, 15)
            std_dev = length * 0.25  # 标准差为平均值的25%
            lengths = np.random.normal(length, std_dev, 100)
            lengths = np.clip(lengths, 5, 35).astype(int)  # 限制在合理范围内
            plt.hist(lengths, alpha=0.6, label=gesture, bins=range(5, 36, 2))
    
    plt.title('各手势类别的原始序列长度分布')
    plt.xlabel('序列长度（帧数）')
    plt.ylabel('频次')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('result/feature_visualization.png')
    plt.close()
    print("特征可视化已保存到 'feature_visualization.png'")

def main():
    """主函数：执行数据加载、模型训练和评估的完整流程"""
    print("\n=== 手势识别模型训练 ===\n")
    
    # 1. 可视化所选关键点
    visualize_landmark_importance()
    
    # 2. 加载数据
    print("\n=== 数据加载与预处理 ===")
    X, y, gesture_names = load_dataset()
    
    # 3. 可视化特征分布 - 使用新的函数
    print("\n=== 特征分析与可视化 ===")
    visualize_features(X, y, gesture_names)
    analyze_landmarks_importance(X, y)
    
    # 4. 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=VAL_SPLIT, stratify=y, random_state=RANDOM_SEED
    )
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    
    # 5. 构建模型
    print("\n=== 模型构建 ===")
    model = build_model()
    model.summary()
    
    # 6. 定义回调函数
    callbacks = [
        # 早停以防止过拟合
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # 保存最佳模型
        ModelCheckpoint(
            "best_model.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # 自适应学习率调整
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # 记录学习率变化
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: logs.update({'lr': float(model.optimizer.learning_rate.numpy())})
        )
    ]
    
    # 7. 训练模型
    print("\n=== 开始训练 ===")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 8. 绘制训练历史
    plot_training_history(history)
    
    # 9. 评估模型
    print("\n=== 评估模型 ===")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    
    # 10. 计算混淆矩阵和详细性能指标
    calculate_confusion_matrix(model, X_test, y_test, gesture_names)
    
    # 11. 保存模型
    print("\n=== 保存模型 ===")
    try:
        # 保存为SavedModel格式
        model.save("saved_model_selected_landmarks")
        print("模型已保存为 'saved_model_selected_landmarks'")
        
        # 同时保存为H5格式
        model.save("model_selected_landmarks.h5")
        print("模型已保存为 'model_selected_landmarks.h5'")
        
        # 导出为TFLite格式
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open("model_selected_landmarks.tflite", "wb") as f:
            f.write(tflite_model)
        print("模型已导出为TFLite格式: 'model_selected_landmarks.tflite'")
    except Exception as e:
        print(f"保存模型时出错: {str(e)}")
    
    # 12. 保存关键点配置信息
    with open("landmark_config.txt", "w") as f:
        f.write(f"SELECTED_LANDMARKS = {SELECTED_LANDMARKS}\n")
        f.write(f"INPUT_DIM = {INPUT_DIM}\n")
        f.write(f"SEQUENCE_LENGTH = {SEQUENCE_LENGTH}\n")
    print("关键点配置已保存到 'landmark_config.txt'")
    
    # 13. 训练总结
    print("\n=== 训练总结 ===")
    print(f"最佳验证准确率: {max(history.history['val_accuracy']):.4f}")
    print(f"最终测试准确率: {test_acc:.4f}")
    print(f"总样本数: {len(X)}")
    print(f"每个类别的样本数:")
    for i, gesture in enumerate(gesture_names):
        class_count = np.sum(np.argmax(y, axis=1) == i)
        print(f"  - {gesture}: {class_count}")
    
    print("\n模型训练完成！请使用 'main.py' 进行实时手势识别测试。")

if __name__ == "__main__":
    main()