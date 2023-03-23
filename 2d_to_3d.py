import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load the 2d image
img = cv.imread("/users/charles/desktop/images/messi_face_2.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# edge detection
edges = cv.Canny(gray, 50, 150)

# depth map generation
disp = cv.StereoSGBM_create().compute(gray, edges)

print(disp)

# 3d model reconstruction
X, Y = np.meshgrid(range(disp.shape[1]), range(disp.shape[0]))

Z = disp

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z)
plt.show()

