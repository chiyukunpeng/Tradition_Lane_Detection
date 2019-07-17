import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

if __name__ == '__main__':
    # 读取图片
    image = cv2.imread('./test_images/1.jpg')
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 多边形角点（左下，左上，右上，右下）
    vertices = np.int32([[(50, 1080),(700, 760),(920, 760),(1400, 1080)]])
    masked_image = region_of_interest(img, vertices)
    cv2.imshow("masked_image", masked_image)
    cv2.waitKey(0)