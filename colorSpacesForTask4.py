import cv2
import skimage.io as io
import numpy as np

if __name__ == '__main__':
	# Read the img, decompose, 
	# apply algotithm for each channel (as greyscale)
	# and merge the result


	# Read
	img = io.imread('in01.jpg')

	# RGB
	b,g,r = cv2.split(img)

	img2 = cv2.merge((b,g,r))
	cv2.imwrite('in02.jpg', img2)

	# YUV
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(img_yuv)

	img2 = cv2.merge((y,u,v))
	cv2.imwrite('in03.jpg', img2)

	# cv.COLOR_BGR2HSV
	img_hsv =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(img_hsv)

	img2 = cv2.merge((h,s,v))
	cv2.imwrite('in04.jpg', img2)

	# cv.COLOR_CMYK
	rgb_scale = 255
	cmyk_scale = 100

	# rgb [0,255] -> cmy [0,1]
	c = 1 - r / 255.
	m = 1 - g / 255.
	y = 1 - b / 255.

	# extract out k [0,1]
	min_cmy = np.min(([c, m, y]))

	c = (c - min_cmy) / (1 - min_cmy)
	m = (m - min_cmy) / (1 - min_cmy)
	y = (y - min_cmy) / (1 - min_cmy)
	k = min_cmy

 	# rescale to the range [0,cmyk_scale]
	c = c*cmyk_scale
	m = m*cmyk_scale
	y = y*cmyk_scale
	k = k*cmyk_scale

	
	# cmyk to rgb
	r = rgb_scale*(1.0-(c+k)/float(cmyk_scale))
	g = rgb_scale*(1.0-(m+k)/float(cmyk_scale))
	b = rgb_scale*(1.0-(y+k)/float(cmyk_scale))
	img2 = cv2.merge((b,g,r))
	cv2.imwrite('in05.jpg', img2)
	

# https://stackoverflow.com/questions/14088375/how-can-i-convert-rgb-to-cmyk-and-vice-versa-in-python



