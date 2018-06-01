import scipy.sparse as sp
import numpy as np

from sklearn.datasets import load_svmlight_file
from os import  listdir
from os.path import basename, join
from Logger.TimeIt import timeit
"""This module incudes common 
utility functions for IO"""

@timeit
def listAllFiles(basePath):
	pathList = []
	for file in listdir(basePath):
		pathList.append(join(basePath, file))
	return pathList

@timeit
def svmLightToNPVectors(pathList, featureCount, dType, includeBias = False):
	csrMatrixList = []
	y = []
	for file in pathList:
		output = load_svmlight_file(file, n_features = featureCount, dtype = dType)
		csrMatrixList.append(output[0])
		y.extend(output[1])
	x = sp.vstack(csrMatrixList)
	if includeBias:
		x = sp.hstack((np.ones((x.shape[0], 1)), x))
	return x.tocsr(), np.array(y).reshape(-1, 1)