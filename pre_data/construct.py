# 子控制文件，用于导入数据集并生成对应的3D数据以及投影之后的2D数据
import numpy as np
import cv2
import matplotlib.image as mpig
import os
from pre_data.NLM import NLM
from pre_data.SSIM import SSIM


def project():
    # 请使用者在此处添加你所使用的数据集的绝对路径
    path = "D:/Filez/DownLoad/大作业/OCTA/OCTA_6M_OCTA"
    save_path = "pre_data/data"
    pack_img = np.arange(10001, 10002, 1)
    div_img = np.arange(1, 401, 1)
    for j in pack_img:
        sum_img = []
        for i in div_img:
            full_path = path + '/' + str(j) + '/' + str(i) + '.bmp'
            print(full_path)
            sum_img.append(mpig.imread(full_path))
        sum_img = np.array(sum_img)
        projection = np.sum(sum_img, axis=1)
        # 将投影转换至uint8
        projection = projection / projection.max()
        projection = np.uint8(projection * 255)
        # 将投影做180度上下翻转至GroundTruth对应空间
        projection = np.flip(projection, 0)
        # 保存原始投影图像
        cv2.imwrite(save_path + "/2D/" + str(j) + ".bmp", projection)
        print(os.path.exists(path))
        projection_nlm = NLM(projection)
        # 保存NLM滤波后投影图像
        cv2.imwrite(save_path + "/2D_NLM/" + str(j) + ".bmp", projection_nlm)
        # projection_ssim = SSIM.SSIM(projection, projection_nlm)
        # 保存SSIM滤波后投影图像
        # cv2.imwrite(save_path+"2D_NLM/" + str(j) + ".bmp", projection_ssim)


if __name__ == "__main__":
    project()
