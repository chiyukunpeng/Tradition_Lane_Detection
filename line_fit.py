import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from region_of_interest import region_of_interest

# 车道线拟合
def line_fit(binary_warped):
	# 直方图
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[100:midpoint]) + 100
	rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint

	# 滑窗数目
	nwindows = 9
	window_height = np.int(binary_warped.shape[0]/nwindows)
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	leftx_current = leftx_base
	rightx_current = rightx_base
	# 滑窗宽度允许变动阈值
	margin = 100
	# 滑窗面积阈值
	minpix = 50

	left_lane_inds = []
	right_lane_inds = []

	for window in range(nwindows):
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height

		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['out_img'] = out_img
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	return ret

# 可视化
def viz2(binary_warped, ret):
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	#滑窗内车道线颜色
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	margin = 100

	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	return result

# 计算曲率半径
def calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy):
	# 图片高度最大值索引
	y_eval = 1079

	# 图片单位像素与米的换算关系
	xm_per_pix = 3.75 / 1100
	ym_per_pix = 30 / 720

	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

	return left_curverad, right_curverad

# 计算距车道中心偏移量
def calc_vehicle_offset(undist, left_fit, right_fit):
	bottom_y = undist.shape[0] - 1
	bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
	bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
	vehicle_offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2
	# 图片单位像素与米的换算关系
	xm_per_pix = 3.75/1100
	vehicle_offset *= xm_per_pix

	return vehicle_offset

# 可视化
def final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset):
	ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# 创建图片
	color_warp = np.zeros((1080, 1920, 3), dtype='uint8')

	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	# 图片右上角显示曲率半径，中心偏移量
	avg_curve = (left_curve + right_curve)/2
	string1 = 'R_mean : %.1f m' % avg_curve
	if left_fit[0] > 0 and avg_curve > 500:
		string2 = "gentle right"
	elif left_fit[0] > 0 and avg_curve <= 500:
		string2 = "hard right"
	elif left_fit[0] < 0 and avg_curve > 500:
		string2 = "gentle left"
	elif left_fit[0] < 0 and avg_curve <= 500:
		string2 = "hard left"
	string3 = 'central offset: %.1f m' % vehicle_offset

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(result, string1, (1500, 100), font, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
	cv2.putText(result, string2, (1500, 300), font, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
	cv2.putText(result, string3, (1500, 200), font, 0.9, (0, 0, 0), 4, cv2.LINE_AA)

	# 图片上方显示消除失真图，鸟瞰图，车道线检测图
	small_undist = cv2.resize(undist, (0, 0), fx=0.2, fy=0.2)
	bird_eye, _, _, _ = perspective_transform(undist)
	small_bird_eye = cv2.resize(bird_eye, (0, 0), fx=0.2, fy=0.2)

	img = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
	vertices = np.int32([[(50, 1080), (700, 760), (920, 760), (1400, 1080)]])
	masked_image = region_of_interest(img, vertices)
	cv2.line(masked_image, (50, 1080), (700, 760), (0, 0, 0), 10)
	cv2.line(masked_image, (920, 760), (1400, 1080), (0, 0, 0), 25)
	img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(masked_image)
	binary_warped, binary_unwarped, m, m_inv = perspective_transform(img)
	ret = line_fit(binary_warped)
	out_image = viz2(binary_warped, ret)
	small_out_image = cv2.resize(out_image, (0, 0), fx=0.2, fy=0.2)

	x1 = 0
	y1 = 100
	x2 = small_out_image.shape[0]
	y2 = small_out_image.shape[1]
	y3 = small_out_image.shape[1] * 2
	y4 = small_out_image.shape[1] * 3
	result[x1 + 100:x2 + 100, y1:y2 + 100, :] = small_undist
	result[x1 + 100:x2 + 100, y2 + 200:y3 + 200, :] = small_bird_eye
	result[x1 + 100:x2 + 100, y3 + 300:y4 + 300, :] = small_out_image
	cv2.putText(result, "Undist", (100, 80), font, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
	cv2.putText(result, "Bird's Eye", (580, 80), font, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
	cv2.putText(result, "Line Search", (1080, 80), font, 0.9, (0, 0, 0), 4, cv2.LINE_AA)

	return result

