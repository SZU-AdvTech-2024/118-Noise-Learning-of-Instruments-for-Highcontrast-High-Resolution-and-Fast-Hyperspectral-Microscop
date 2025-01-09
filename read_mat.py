# import scipy.io
# import matplotlib.pyplot as plt
#
# # 加载.mat文件
# data = scipy.io.loadmat(r'D:\研一\培养计划\Noise-learning-1.0\data\NoiseDataforHoribaLabRAM\n0.1s.mat')
#
# # 提取cube属性的数据
# cube_data = data['predr']
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# # 加载.mat文件中的数据
# data = loadmat(r'D:\研一\培养计划\Noise-learning-1.0\data\NoiseDataforHoribaLabRAM\n0.1s.mat')  # 替换为你的.mat文件路径
# cube = data['cube']
# cube = cube.T
# x = data['x'].flatten()  # 确保x是一维数组
# noise = data['noise']
# noise = noise.T
#
# # 可视化第一个光谱样本
# plt.figure(figsize=(12, 6))
#
# # 绘制原始光谱
# plt.subplot(2, 1, 1)
# plt.plot(x, cube[0, :])  # 假设我们可视化第一个样本
# plt.title('Raman Spectrum')
# plt.xlabel('Wavenumber (cm⁻¹)')
# plt.ylabel('Intensity')
#
# # 绘制噪声
# plt.subplot(2, 1, 2)
# plt.plot(x, noise[0, :])  # 假设我们可视化第一个样本的噪声
# plt.title('Instrumental Noise')
# plt.xlabel('Wavenumber (cm⁻¹)')
# plt.ylabel('Noise Intensity')
#
# plt.tight_layout()
# plt.show()


# # 选取100个数据点
# num_points = 10
# selected_indices = np.random.choice(2600, num_points, replace=False)
#
# # 绘制图形
# plt.figure(figsize=(15, 7))  # 设置图形大小
#
# for i in selected_indices:
#     plt.plot(noise_data[:, i], label=f'Data Point {i+1}')

import numpy as np
import matplotlib.pyplot as plt

# 假设 data 是一个 (2500, 1600) 的 NumPy 数组，代表你的拉曼数据

data = loadmat(r'D:\研一\培养计划\Noise-learning-1.0\data\NoiseDataforHoribaLabRAM\n0.1s.mat')  # 替换为你的.mat文件路径
noise_data = data['predr']

plt.plot(noise_data[:, 763], label=f'Data Point')

# 设置横坐标和纵坐标的标签
plt.xlabel('Band Number')
plt.ylabel('Value')

# 显示图例
plt.legend()

# 显示图形
plt.show()