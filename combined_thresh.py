import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# SobelX，SobelY梯度信息
def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	binary_output = np.zeros_like(scaled_sobel)
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

	return binary_output

# SobelXY梯度信息
def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	scale_factor = np.max(gradmag)/255
	gradmag = (gradmag/scale_factor).astype(np.uint8)
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

	return binary_output

# direction信息
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

	return binary_output

# S通道信息
def hls_thresh(img, thresh=(100, 255)):
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	binary_output = np.zeros_like(s_channel)
	binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

	return binary_output

# 叠加信息，四组阈值可以适当调整
def combined_thresh(img):
	abs_bin = abs_sobel_thresh(img, orient='x', thresh_min=35, thresh_max=100)
	mag_bin = mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 255))
	dir_bin = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
	hls_bin = hls_thresh(img, thresh=(80, 255))

	combined = np.zeros_like(dir_bin)
	combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | hls_bin == 1] = 1

	return combined, abs_bin, mag_bin, dir_bin, hls_bin


if __name__ == '__main__':

	img_file = 'test_images/1.jpg'

	with open('calibrate_camera.p', 'rb') as f:
		save_dict = pickle.load(f)
	mtx = save_dict['mtx']
	dist = save_dict['dist']
	# 读取图片
	img = mpimg.imread(img_file)

	img = cv2.undistort(img, mtx, dist, None, mtx)

	combined, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img)

	plt.subplot(2, 3, 1)
	plt.imshow(abs_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(2, 3, 2)
	plt.imshow(mag_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(2, 3, 3)
	plt.imshow(dir_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(2, 3, 4)
	plt.imshow(hls_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(2, 3, 5)
	plt.imshow(img)
	plt.subplot(2, 3, 6)
	plt.imshow(combined, cmap='gray', vmin=0, vmax=1)

	plt.tight_layout()
	plt.show()
