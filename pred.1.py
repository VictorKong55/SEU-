import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from keras.models import load_model

# 读取 .mat 文件
data = scipy.io.loadmat('data_pred1.mat')

# 提取数据
F = data['F']        # 载荷数据 (8,1000)
time = data['time']  # 时间数据 (1,1000)
x = data['x']        # 位移数据 (8,1000)

F_input_data = F/1000  # (8,1000)
x_output_data = x*1000  # (8,1000)

# 整形数据
F_reshaped = np.expand_dims(F_input_data, axis=-1)  # (8, 1000, 1)
x_reshaped = np.expand_dims(x_output_data, axis=-1)  # (8, 1000, 1)

# 加载已训练的模型
model = load_model('my_model.keras')
model.compile(optimizer='adam', loss='mse')

predictions = model.predict(F_reshaped)

plt.subplot(2,2,1)
plt.plot(time[0,], x_output_data[4,], color='red', label='True x ')
plt.plot(time[0,], predictions[4,:,0], color='blue', label='Predicted x ')
plt.subplot(2,2,2)
plt.plot(time[0,], x_output_data[5,], color='red', label='True x ')
plt.plot(time[0,], predictions[5,:,0], color='blue', label='Predicted x ')
plt.subplot(2,2,3)
plt.plot(time[0,], x_output_data[6,], color='red', label='True x ')
plt.plot(time[0,], predictions[6,:,0], color='blue', label='Predicted x ')
plt.subplot(2,2,4)
plt.plot(time[0,], x_output_data[7,], color='red', label='True x ')
plt.plot(time[0,], predictions[7,:,0], color='blue', label='Predicted x ')
plt.legend()
plt.show()
