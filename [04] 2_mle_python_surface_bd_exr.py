# import modules
import numpy as np
import matplotlib.pyplot as plt

N = 50
S = np.arange(1,N+1)
theta = np.linspace(0.1,0.9,100)

S_grid, theta_grid = np.meshgrid(S, theta)
L = S_grid*np.log(theta_grid) + (N-S_grid)*np.log(1-theta_grid)

S_index = np.where(S == 12)[0][0]
max_likelihood_index = np.unravel_index(np.argmax(L[S_index]), L.shape)
max_likelihood_theta = theta[max_likelihood_index[1]]

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
s = ax.plot_surface(S_grid, theta_grid, L, cmap='jet')
ax.set_xlabel('S')
ax.set_ylabel('theta')
ax.set_zlabel('L(theta|S)')
ax.set_title('Maximum Likelihood Estimation')
ax.view_init(65,15)

ax.scatter(12,max_likelihood_theta, L[S_index, max_likelihood_index[1]], color='red', s=100, label='MLE')
ax.legend()
# plt.show()
plt.savefig("./pictures/s-theta-L-12.png")