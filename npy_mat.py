import numpy as np
import scipy.io

# 加载 NPY 文件
data = np.load("Chikusei_gt.npy")
print(data.shape)

# 保存为 MAT 文件
scipy.io.savemat("chikusei.mat", {"gt": data[0]})
