import numpy as np

from clase_2.synthetic_dataset import SyntheticDataset
from clase_2.random_variable import exponential_random_variable
from clase_2.k_means import k_means


# exercice # 1

X = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
C = np.array([
    [1,0,0],
    [0,1,1]
])
expanded_C = C[:, None]
distances = np.sqrt(np.sum((expanded_C - X) ** 2, axis=2))
print(distances)
# [[ 3.60555128  8.36660027 13.45362405]
#  [ 2.44948974  7.54983444 12.72792206]]


# exercice # 8
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
x = np.array([ [0.4, 4800, 5.5], [0.7, 12104, 5.2], [1, 12500, 5.5], [1.5, 7002, 4.0] ])
pca = PCA(n_components=3)
x_std = StandardScaler(with_std=False).fit_transform(x)
pca.fit_transform(x_std)


x2 = (x - x.mean(axis=0))
cov_1 = np.cov(x2.T)
w, v = np.linalg.eig(cov_1)
idx = w.argsort()[::-1]
w = w[idx]
v = v[:,idx]
np.matmul(x2, v[:, :2])


# exercise 12

# create the synthethic dataset
synthetic_dataset = SyntheticDataset(n_samples=10000, inv_overlap=10)
train, train_cluster_ids, valid, valid_cluster_ids = synthetic_dataset.train_valid_split()

# apply pca over train data to plot the cluster
train_pca = SyntheticDataset.reduce_dimension(train, 2)
SyntheticDataset.plot_cluster(train_pca, train_cluster_ids)

# expand the train data with one extra feature (exponential random variable)
exponential_rv = exponential_random_variable(lambda_param=1, size=train.shape[0])
train_expanded = np.concatenate((train, exponential_rv.reshape(-1,1)), axis=1)

# apply pca over the extended train data to plot the cluster
train_expanded_pca = SyntheticDataset.reduce_dimension(train_expanded, 2)
SyntheticDataset.plot_cluster(train_expanded_pca, train_cluster_ids)

# use k-means to cluster the data and the plot the new cluster
centroids, cluster_ids = k_means(train_expanded, 2)
SyntheticDataset.plot_cluster(train_expanded_pca, cluster_ids)
