import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage import median_filter
import glob
import numpy as np
import sys
import argparse
import cv2

from MyKmeans4 import MyKmeans
from MyGMM4 import MyGMM
from MyFCM4 import MyFCM
from MySOM4 import MySOM
from MySpectral4 import MySpectral


algos = ["Kmeans", "SOM", "FCM", "Spectral", "GMM"]
types = ["RGB", "Hyper"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="image cluster")
    parser.add_argument(
        "--algo",
        type=str,
        help="pleaser choose from [Kmeans, SOM, FCM, Spectral, GMM]",
        default="Spectral",
    )
    parser.add_argument(
        "--img_type", type=str, help="please choose from [RGB, Hyper]", default="Hyper"
    )
    parser.add_argument("--num_cluster", type=int, help="number of cluster", default=20)
    parser.add_argument("--mat_path", type=str, help="image path", default="pavia/PaviaU.mat")

    args = parser.parse_args()
    print(args.mat_path)
    mat = sio.loadmat(args.mat_path)
    # img = mat["gt"].transpose(1, 2, 0)
    # img = mat["gt"]   # chikusei
    img = mat["paviaU"] # Pavia
    array_len = img.shape[0] * img.shape[1] * img.shape[2]
    print(img.shape)

    if args.num_cluster == 1:
        args.num_cluster = int(0.05 * array_len)
    elif args.num_cluster > array_len / 4:
        args.num_cluster = array_len / 4

    assert args.algo in algos
    assert args.img_type in types

    option = algos.index(args.algo)

    if option == 0:
        result = MyKmeans(img, args.img_type, args.num_cluster)
    elif option == 1:
        result = MySOM(img, args.img_type, args.num_cluster)
    elif option == 2:
        result = MyFCM(img, args.img_type, args.num_cluster)
    elif option == 3:
        result = MySpectral(img, args.img_type, args.num_cluster)
    elif option == 4:
        result = MyGMM(img, args.img_type, args.num_cluster)

    print(args.img_type)
    if args.img_type == "RGB":
        conn_comp = result[1]
        result = result[0]
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        img1 = ax1.imshow(result)
        fig.colorbar(img1, ax=ax1)
        ax1.set_title("Clustered Image")
        ax1.set_aspect("auto")
        img2 = ax2.imshow(conn_comp)
        fig.colorbar(img2, ax=ax2)
        ax2.set_title("Connected Components")
        ax2.set_aspect("auto")
        plt.tight_layout(h_pad=1)

    else:
        plt.imshow(result)
        plt.title("Clustered Image")
        plt.colorbar()

    # cv2.imshow("origin image", img[:, :, 10])

    plt.show()
    sio.savemat("pavia/PaviaU_res.mat", {"res" : result})