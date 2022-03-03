# 主控制文件，在此处可直接利用测试组测试网络性能
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import Args
from skimage import measure, morphology
from skimage.measure import _marching_cubes_classic
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    p = p[:, :, ::-1]
    print(p.shape)
    mem = measure.marching_cubes(p, threshold)
    print(mem)
    print(mem[0].shape)
    print(mem[1].shape)
    print(mem[2].shape)
    print(mem[3].shape)
    verts = np.array(mem[0])
    faces = np.array(mem[1])
    faces.astype(np.int16)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    poly3d = [[verts[faces[ix][iy]] for iy in range(len(faces[0]))]
              for ix in range(len(faces))]
    # mesh = Poly3DCollection(verts[faces], alpha=1)
    ax.scatter(verts[0], verts[1], verts[2])
    # ax.add_collection3d(Poly3DCollection(poly3d, facecolors='w', linewidths=1, alpha=0.5))
    # face_color = [0.5, 0.5, 1]
    # mesh.set_facecolor(face_color)
    # ax.add_collection3d(mesh)
    #
    # ax.set_xlim(0, p.shape[0])
    # ax.set_ylim(0, p.shape[1])
    # ax.set_zlim(0, p.shape[2])

    plt.show()


path = Args.OCTA_path + '/10001/'
div_img = np.arange(1, 401, 1)
img = []
for i in div_img:
    full_path = path + str(i) + '.bmp'
    print(full_path)
    img.append(cv2.imread(full_path, 0))
img = np.array(img)
# img = cv2.imread(path+str(1)+'.bmp')
img.resize(640, 400, 400);
plot_3d(img, 100)
