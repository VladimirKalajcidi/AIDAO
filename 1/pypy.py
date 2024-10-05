import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
# from sklearn.cluster import KMeans


# data_o = np.load("data/ihb.npy")
# data_o = np.nan_to_num(data_o, nan=-15)
# data_n = np.zeros((320, 246, 55))



# data_n[0, 0, :10] = data_o[0, :, 0]
# last = 10
# for i in range(1, 10):
# 	data_n[0, 0, last:last+10-i] = data_o[0, i:, 0] - data_o[0, :-i, 0]
# 	last += 10
# 	last -= i
	

# print(data_o[0, :, 0])
# print(data_n[0, 0, :])

# # print(data_n.shape)

# # estimators = KMeans(n_clusters=20, n_init=5)


# # estimators.fit(data_n)
# # labels = estimators.labels_
# # pd.DataFrame({'prediction': labels}).to_csv('mysubmission.csv', index=False)
# x, y = np.random.rand(2, 100) * 4
# hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])
# print(yedges)

import matplotlib.pyplot as plt
import numpy as np

# # Fixing random state for reproducibility
# np.random.seed(19680801)

# z = np.array([[1, 1, 2], [2, 2, 3], [4, 4, 6]])

# x = np.array([[i] * 3 for i in range(3)]).ravel() # x coordinates of each bar
# y = np.array([i for i in range(3)] * 3) # y coordinates of each bar
# z = np.zeros(9) # z coordinates of each bar
# dx = np.ones(9) # length along x-axis of each bar
# dy = np.ones(9) # length along y-axis of each bar
# dz = z.ravel() # length along z-axis of each bar (height)
# # y = np.array([i for i in range(3)] * 3)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # x, y = np.random.rand(2, 100) * 4
# # hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])

# # Construct arrays for the anchor positions of the 16 bars.
# # xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
# # x = x.ravel()
# # y = y.ravel()
# # # print(xpos.shape, ypos.shape)
# # zpos = 0

# # # Construct arrays with the dimensions for the 16 bars.
# # dx = dy = 0.5 * np.ones_like(zpos)
# dz = 0
# # print(dx, dy, dz)
# # print(zpos)

# ax.bar3d(dx, dy, dz, x, y, z, zsort='average')

# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# # Fixing random state for reproducibility
# np.random.seed(19680801)


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# np.random.seed(1234)
# A = np.random.randint(5, size=(5, 5))

# x = np.array([[i] * 5 for i in range(5)]).ravel() # x coordinates of each bar
# y = np.array([i for i in range(5)] *5) # y coordinates of each bar
# z = np.zeros(25) # z coordinates of each bar
# dx = np.ones(25) # length along x-axis of each bar
# dy = np.ones(25) # length along y-axis of each bar
# dz = A.ravel() # length along z-axis of each bar (height)

# ax.bar3d(x, y, z, dx, dy, dz, zsort='average')

# plt.show()

import matplotlib.pyplot as plt
import numpy as np

N = 3
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
hist = np.array([[1, 2, 2], [2, 3, 3], [4, 6, 6]])
xpos, ypos = np.meshgrid(np.arange(N+1)[:-1] + 0.25, np.arange(N+1)[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

print(xpos, ypos, zpos, dx, dy, dz)
print(zpos, dx, dy)
print(dz)
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.show()