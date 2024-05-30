import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from keras.models import load_model

# 读取 .mat 文件
data = scipy.io.loadmat('data_pred2.mat')

# 提取数据
F = data['F']        # 载荷数据 (2,1000)
time = data['time']  # 时间数据 (1,1000)
x = data['x']        # 位移数据 (2,1000)

F_input_data = F/1000  # (2,1000)
x_output_data = x*1000  # (2,1000)

# 整形数据
F_reshaped = np.expand_dims(F_input_data, axis=-1)  # (2, 1000, 1)
x_reshaped = np.expand_dims(x_output_data, axis=-1)  # (2, 1000, 1)

# 加载已训练的模型
model = load_model('my_model.keras')
model.compile(optimizer='adam', loss='mse')

predictions = model.predict(F_reshaped)

plt.subplot(2,2,1)
plt.plot(time[0,], x_output_data[0,], color='red', label='True x ')
plt.plot(time[0,], predictions[0,:,0], color='blue', label='Predicted x ')
plt.subplot(2,2,2)
plt.plot(time[0,], x_output_data[1,], color='red', label='True x ')
plt.plot(time[0,], predictions[1,:,0], color='blue', label='Predicted x ')
plt.subplot(2,2,3)
plt.plot(time[0,], F_input_data[0,], color='red', label='True x ')
plt.subplot(2,2,4)
plt.plot(time[0,], F_input_data[1,], color='red', label='True x ')
plt.legend()
plt.show()