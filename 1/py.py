import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import rand_score
from sklearn.preprocessing import StandardScaler


data = np.load("data/ihb.npy")
mask = np.isnan(data).any(axis=1).any(axis=1)

data_1 = data[np.array(mask)] # with Nan
data_2 = data[~np.array(mask)] # without Nan

good_features_1 = [0, 2, 3, 5, 7, 11, 13, 14, 15, 16, 17, 18, 21, 22, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 106, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 130, 132, 135, 136, 137, 138, 139, 141, 142, 143, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 160, 161, 162, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198]
good_features_2 = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 29, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 48, 49, 50, 51, 54, 55, 56, 58, 60, 61, 63, 64, 65, 66, 67, 68, 69, 71, 73, 74, 76, 77, 79, 80, 82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 96, 98, 100, 102, 104, 105, 106, 108, 111, 113, 115, 116, 117, 121, 123, 124, 125, 126, 127, 129, 130, 132, 134, 136, 137, 138, 139, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 184, 185, 186, 187, 194, 195, 198, 199, 200, 203, 208, 209, 210, 211, 212, 213, 214, 215, 217, 218, 219, 220, 221, 222, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 237, 238, 239, 240, 241, 243, 244, 245]

labels_1 = np.zeros((160, len(good_features_1)))
labels_2 = np.zeros((160, len(good_features_2)))

for i in range(len(good_features_1)):
	data = StandardScaler().fit_transform(data_1[:, :, good_features_1[i]])
	labels_1[:, i] = KMeans(n_clusters=20).fit(data).labels_

for i in range(len(good_features_2)):
	data = StandardScaler().fit_transform(data_2[:, :, good_features_2[i]])
	labels_2[:, i] = KMeans(n_clusters=20).fit(data).labels_

final_1 = KMeans(n_clusters=20).fit(labels_1).labels_
final_2 = KMeans(n_clusters=20).fit(labels_2).labels_

answer = np.zeros(320, dtype=int)

i_1 = 0
i_2 = 0
for i in mask:
	if i:
		answer[i_1 + i_2] = final_1[i_1]
		i_1 += 1
	else:
		answer[i_1 + i_2] = final_2[i_2]
		i_2 += 1


pd.DataFrame({'prediction': answer}).to_csv('submission.csv', index=False)