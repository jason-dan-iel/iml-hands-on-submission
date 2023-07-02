import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

lambd = cv2.imread('galaxy.tif')
lambd = cv2.cvtColor(lambd, cv2.COLOR_BGR2GRAY)/255
T = 100
lambdT = np.repeat(lambd[:, :, np.newaxis], T, axis = 2)
x = stats.poisson.rvs(lambdT)
y = (x>=1).astype(float)
lambdhat = -np.log(1-np.mean(y,axis=2))
plt.imshow(lambdhat, cmap='gray')
# plt.show()
plt.savefig('./pictures/mle-galaxy.png')