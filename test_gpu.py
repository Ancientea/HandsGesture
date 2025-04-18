import tensorflow as tf

print("TensorFlow版本:", tf.__version__)
print("GPU是否可用:", tf.config.list_physical_devices('GPU'))
print("CUDA是否可用:", tf.test.is_built_with_cuda()) 