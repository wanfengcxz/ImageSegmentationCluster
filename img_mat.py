from PIL import Image
import numpy as np
import scipy.io

# 打开 JPG 图像
image = Image.open("./img/img2.jpg")

# 将图像转换为 NumPy 数组
image_array = np.array(image)

# 保存为 MAT 文件
scipy.io.savemat("image2.mat", {"img": image_array})
