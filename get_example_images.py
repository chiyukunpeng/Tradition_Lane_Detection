import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from line_fit import line_fit, calc_curve, final_viz
from region_of_interest import region_of_interest

with open('calibrate_camera.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

# 批量读取图片
image_files = os.listdir('test_images')
for image_file in image_files:
	out_image_file = image_file.split('.')[0] + '.png'  # write to png format
	img = mpimg.imread('test_images/' + image_file)

	# 校正图片
	img = cv2.undistort(img, mtx, dist, None, mtx)
	plt.imshow(img)
	plt.savefig('example_images/undistort' + out_image_file)

	# ROI操作
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# 多边形角点（左下，左上，右上，右下）
	vertices = np.int32([[(50, 1080), (700, 760), (920, 760), (1400, 1080)]])
	masked_image = region_of_interest(img, vertices)
	# 消除ROI边界影响，宽度适当调整
	cv2.line(masked_image, (50, 1080), (700, 760), (0, 0, 0),10)
	cv2.line(masked_image, (920, 760), (1400, 1080), (0, 0, 0),25)

	# 二值化
	img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(masked_image)
	plt.imshow(img, cmap='gray', vmin=0, vmax=1)
	plt.savefig('example_images/binary' + out_image_file)

	# 透视变换
	img, binary_unwarped, m, m_inv = perspective_transform(img)
	plt.imshow(img, cmap='gray', vmin=0, vmax=1)
	plt.savefig('example_images/warped' + out_image_file)

	# 多项式拟合
	ret = line_fit(img)
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']
	save_file = 'example_images/polyfit' + out_image_file


	# 批量读取图片
	orig = mpimg.imread('test_images/' + image_file)
	undist = cv2.undistort(orig, mtx, dist, None, mtx)
	left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
	# 图片单位像素与米的换算关系
	xm_per_pix = 3.75 / 1100

	bottom_y = undist.shape[0] - 1
	bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
	bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
	vehicle_offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2
	vehicle_offset *= xm_per_pix

	img = final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)
	plt.imshow(img)
	plt.savefig('example_images/annotated' + out_image_file)

