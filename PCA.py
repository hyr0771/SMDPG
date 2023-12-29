import numpy as np
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, IncrementalPCA
from sklearn import preprocessing

def getPCA_feature(X, num):
	X = preprocessing.scale(X)

	pca = PCA(n_components=num)
	pca.fit(X)

	X_PCA = pca.transform(X)
	print(pca.transform(X))

	print('主成分系数矩阵是:', pca.components_)
	print('特征值:', pca.explained_variance_)
	print('⽅差解释率', pca.explained_variance_ratio_)
	ans = pca.explained_variance_ratio_
	print('sum: ', sum(ans))

	return X_PCA

def other_PCA(X, num):
	# X = preprocessing.scale(X)
	pca = IncrementalPCA(n_components=num, batch_size=1024)
	pca.fit(X)

	X_PCA = pca.transform(X)
	print(pca.transform(X))

	print('主成分系数矩阵是:', pca.components_)
	print('特征值:', pca.explained_variance_)
	print('⽅差解释率', pca.explained_variance_ratio_)
	ans = pca.explained_variance_ratio_
	print('sum: ', sum(ans))
	return X_PCA

if __name__ == "__main__":
	X = np.loadtxt('X.csv', delimiter=',')
	print("X shape: ", X.shape)
	X_PCA = getPCA_feature(X, 512)
	np.savetxt("X_PCA512.csv", X_PCA, fmt="%.8f", delimiter=",")
	print("run PCA OK......")