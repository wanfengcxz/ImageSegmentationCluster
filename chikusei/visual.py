import cv2
import scipy.io


def Chikusei():
    # 读取 .mat 文件
    mat_data = scipy.io.loadmat('chikusei.mat')
    print(mat_data)

    # 访问数据
    data = mat_data['gt'][:,:,20:23]

    print(data.shape)

    data = data / data.max()
    cv2.imshow("gt", data)
    cv2.waitKey(0)

if __name__ == "__main__":
    # PaviaU()
    # PaviaRes()
    Chikusei()