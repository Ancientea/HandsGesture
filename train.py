import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 超参数配置
DATA_DIR = "gesture_data"
SEQUENCE_LENGTH = 30  # 30帧
INPUT_DIM = 63  # 21关键点 × 3坐标
NUM_CLASSES = len(os.listdir(DATA_DIR))  # 手势类别数
BATCH_SIZE = 32
EPOCHS = 50


def load_dataset():
    X, y = [], []
    for label_idx, gesture in enumerate(os.listdir(DATA_DIR)):
        gesture_dir = os.path.join(DATA_DIR, gesture)
        for file in os.listdir(gesture_dir):
            data = np.load(os.path.join(gesture_dir, file))
            X.append(data)
            y.append(label_idx)

    X = np.array(X)
    y = to_categorical(y, num_classes=NUM_CLASSES)
    return X, y


def build_model():
    inputs = Input(shape=(SEQUENCE_LENGTH, INPUT_DIM))

    # 模型结构：BiLSTM + Transformer (可选)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(32))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu')(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    # 加载数据
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # 数据增强：添加高斯噪声
    noise_factor = 0.01
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_train = np.concatenate([X_train, X_train_noisy])
    y_train = np.concatenate([y_train, y_train])

    # 构建并训练模型
    model = build_model()
    model.summary()

    # 定义回调
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
    ]

    # 训练
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    # 保存模型结构和权重
    model_json = model.to_json()
    with open("model_architecture.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_weights.h5")

    # 评估
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"测试准确率: {test_acc:.4f}")