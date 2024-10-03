import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd

data = np.load("data/ihb.npy")
common_brain_region_data = data[:, :, 0]

model = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=20)
)

model.fit(common_brain_region_data)

cluster_distances = model.transform(common_brain_region_data)

labeling = np.zeros(len(data), dtype=int)
leftover_indexes = np.arange(len(data))
for i in range(20):
    distances_from_current_cluster_center = cluster_distances[:, i]
    if len(distances_from_current_cluster_center) > 16:
        top16 = np.argpartition(distances_from_current_cluster_center, 16)[:16]
        labeling[leftover_indexes[top16]] = i
        cluster_distances = np.delete(cluster_distances, top16, axis=0)
        leftover_indexes = np.delete(leftover_indexes, top16)
    else:
        labeling[leftover_indexes] = i

pd.DataFrame({'prediction': labeling}).to_csv('submission.csv', index=False)