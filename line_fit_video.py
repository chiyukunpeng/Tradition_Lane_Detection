import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from Line import Line
from line_fit import line_fit,final_viz, calc_curve, calc_vehicle_offset
from moviepy.editor import VideoFileClip
from region_of_interest import region_of_interest

with open('calibrate_camera.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

window_size = 5
left_line = Line(n=window_size)
right_line = Line(n=window_size)
detected = False
left_curve, right_curve = 0., 0.
left_lane_inds, right_lane_inds = None, None


def annotate_image(img_in):
	global mtx, dist, left_line, right_line, detected
	global left_curve, right_curve, left_lane_inds, right_lane_inds

	# 校正图片
	undist = cv2.undistort(img_in, mtx, dist, None, mtx)

	# ROI操作
	img = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
	# 多边形角点（左下，左上，右上，右下）
	vertices = np.int32([[(50, 1080), (700, 760), (920, 760), (1400, 1080)]])
	masked_image = region_of_interest(img, vertices)
	# 消除ROI边界影响，宽度适当调整
	cv2.line(masked_image, (50, 1080), (700, 760), (0, 0, 0), 10)
	cv2.line(masked_image, (920, 760), (1400, 1080), (0, 0, 0), 25)

	# 二值化
	img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(masked_image)

	# 透视变换
	binary_warped, binary_unwarped, m, m_inv = perspective_transform(img)

	# 多项式拟合
	if not detected:
		ret = line_fit(binary_warped)
		left_fit = ret['left_fit']
		right_fit = ret['right_fit']
		nonzerox = ret['nonzerox']
		nonzeroy = ret['nonzeroy']
		left_lane_inds = ret['left_lane_inds']
		right_lane_inds = ret['right_lane_inds']

		left_fit = left_line.add_fit(left_fit)
		right_fit = right_line.add_fit(right_fit)

		# 计算曲率半径
		undist = cv2.undistort(img_in, mtx, dist, None, mtx)
		left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
	# 计算距车道中心偏移量
	vehicle_offset = calc_vehicle_offset(undist, left_fit, right_fit)
	# 可视化
	result = final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)

	return result


def annotate_video(input_file, output_file):
	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(annotate_image)
	annotated_video.write_videofile(output_file, audio=False)


if __name__ == '__main__':
	# 输入输出视频
	annotate_video('test.mp4', 'result_out1.mp4')

