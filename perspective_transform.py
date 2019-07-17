import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh


def perspective_transform(img):
	img_size = (img.shape[1], img.shape[0])

    # 源点（左下，左上，右上，右下）
	src = np.float32([[310, 1080],[750, 760],[910, 760],[1370, 1080]])
	# 目标点（左下，左上，右上，右下）
	dst = np.float32([[500,1080],[500,0],[1300,0], [1300,1080]])

	# # 备份demo对应点
	# src = np.float32([[200, 720],[1100, 720],[595, 450],[685, 450]])
	# dst = np.float32([[300, 720],[980, 720],[300, 0],[980, 0]])

	# 透射变换矩阵，逆透视变换矩阵
	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)

	warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
	unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)

	return warped, unwarped, m, m_inv


if __name__ == '__main__':
	img_file = 'test_images/1.jpg'

	with open('calibrate_camera.p', 'rb') as f:
		save_dict = pickle.load(f)
	mtx = save_dict['mtx']
	dist = save_dict['dist']
	# 读取图片
	img = mpimg.imread(img_file)
	img = cv2.undistort(img, mtx, dist, None, mtx)

	img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img)

	warped, unwarped, m, m_inv = perspective_transform(img)
	cv2.imshow('warped.jpg',warped)
	cv2.imshow('unwarped.jpg',unwarped )
	cv2.waitKey(0)

