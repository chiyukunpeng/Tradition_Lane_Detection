import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


class Line():
	def __init__(self, n):
		self.n = n
		self.detected = False

		# 多项式拟合方程系数
		self.A = []
		self.B = []
		self.C = []

		self.A_avg = 0.
		self.B_avg = 0.
		self.C_avg = 0.

	def get_fit(self):
		return (self.A_avg, self.B_avg, self.C_avg)

	def add_fit(self, fit_coeffs):
		q_full = len(self.A) >= self.n

		self.A.append(fit_coeffs[0])
		self.B.append(fit_coeffs[1])
		self.C.append(fit_coeffs[2])

		if q_full:
			_ = self.A.pop(0)
			_ = self.B.pop(0)
			_ = self.C.pop(0)

		self.A_avg = np.mean(self.A)
		self.B_avg = np.mean(self.B)
		self.C_avg = np.mean(self.C)

		return (self.A_avg, self.B_avg, self.C_avg)

