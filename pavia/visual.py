import cv2
import scipy.io


def PaviaGT():
    # 读取 .mat 文件
    mat_data = scipy.io.loadmat('PaviaU_gt.mat')
    print(mat_data)

    # 访问数据
    data = mat_data['paviaU_gt']

    print(data.shape)

    data = data / data.max()
    cv2.imshow("gt", data)
    cv2.waitKey(0)

#
# def PaviaU():
#     # 读取 .mat 文件
#     mat_data = scipy.io.loadmat('PaviaU.mat')
#     print(mat_data)
#
#     # 访问数据
#     data = mat_data['paviaU']
#
#     print(data.shape)
#
#     data = data[:, :, 24]
#     data = data / data.max()
#     cv2.imshow("gt", data)
#     cv2.waitKey(0)
#
#
# def PaviaRes():
#     mat_data = scipy.io.loadmat('PaviaU_res.mat')
#     print(mat_data)
#
#     # 访问数据
#     data = mat_data['res']
#
#     print(data.shape)
#
#     data = data / data.max()
#     cv2.imshow("gt", data)
#     cv2.waitKey(0)
#
#
if __name__ == "__main__":
    # PaviaU()
    # PaviaRes()
    PaviaGT()