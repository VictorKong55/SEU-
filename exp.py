import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

# 读取 .mat 文件
data = scipy.io.loadmat('data_exp.mat')

# 提取数据
F = data['F']       # 载荷数据 (120,1000)
time = data['time']  # 时间数据 (1,1000)
x = data['x']       # 位移数据 (120,1000)

F_input_data = F/1000   # (120,1000)
x_output_data = x*1000  # (120,1000)

# 超参数设定
TIME_STEP = 1000       # 时间步长
INPUT_SIZE = 1         # 输入特征的数量
NUM_EPOCHS = 100       # 迭代次数
BATCH_SIZE = 12        # 批量大小

# 整形数据
F_input_data_reshaped = np.expand_dims(F_input_data, axis=-1)  # (120, 1000, 1)
x_output_data_reshaped = np.expand_dims(x_output_data, axis=-1)  # (120, 1000, 1)

# 构建模型
model = Sequential()
model.add(LSTM(units=128, input_shape=(TIME_STEP, INPUT_SIZE), return_sequences=True))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dense(units=32))
model.add(Dense(units=1))
model.compile(optimizer=Adam(clipnorm=1.0), loss='mean_squared_error')
model.summary()

# 定义 ReduceLROnPlateau 回调函数
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=0.0001)

# 训练模型
history = model.fit(F_input_data_reshaped, x_output_data_reshaped, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[reduce_lr])
predictions = model.predict(F_input_data_reshaped)

# 绘制训练损失
plt.plot(history.history['loss'], label='Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.subplot(3,4,1)
plt.plot(time[0,], x_output_data[0,], color='red', label='True x ')
plt.plot(time[0,], predictions[0,:,0], color='blue', label='Predicted x ')
plt.subplot(3,4,2)
plt.plot(time[0,], x_output_data[1,], color='red', label='True x ')
plt.plot(time[0,], predictions[1,:,0], color='blue', label='Predicted x ')
plt.subplot(3,4,3)
plt.plot(time[0,], x_output_data[2,], color='red', label='True x ')
plt.plot(time[0,], predictions[2,:,0], color='blue', label='Predicted x ')
plt.subplot(3,4,4)
plt.plot(time[0,], x_output_data[3,], color='red', label='True x ')
plt.plot(time[0,], predictions[3,:,0], color='blue', label='Predicted x ')
plt.subplot(3,4,5)
plt.plot(time[0,], x_output_data[4,], color='red', label='True x ')
plt.plot(time[0,], predictions[4,:,0], color='blue', label='Predicted x ')
plt.subplot(3,4,6)
plt.plot(time[0,], x_output_data[5,], color='red', label='True x ')
plt.plot(time[0,], predictions[5,:,0], color='blue', label='Predicted x ')
plt.subplot(3,4,7)
plt.plot(time[0,], x_output_data[6,], color='red', label='True x ')
plt.plot(time[0,], predictions[6,:,0], color='blue', label='Predicted x ')
plt.subplot(3,4,8)
plt.plot(time[0,], x_output_data[7,], color='red', label='True x ')
plt.plot(time[0,], predictions[7,:,0], color='blue', label='Predicted x ')
plt.subplot(3,4,9)
plt.plot(time[0,], x_output_data[8,], color='red', label='True x ')
plt.plot(time[0,], predictions[8,:,0], color='blue', label='Predicted x ')
plt.subplot(3,4,10)
plt.plot(time[0,], x_output_data[9,], color='red', label='True x ')
plt.plot(time[0,], predictions[9,:,0], color='blue', label='Predicted x ')
plt.subplot(3,4,11)
plt.plot(time[0,], x_output_data[10,], color='red', label='True x ')
plt.plot(time[0,], predictions[10,:,0], color='blue', label='Predicted x ')
plt.subplot(3,4,12)
plt.plot(time[0,], x_output_data[11,], color='red', label='True x ')
plt.plot(time[0,], predictions[11,:,0], color='blue', label='Predicted x ')
plt.legend()
plt.show()
# 保存模型
model.save('my_model.keras')
