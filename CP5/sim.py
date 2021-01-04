import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist,squareform

# test 
# a = np.array([0,0,0,1,1,1,1])
# b = np.arange(15).reshape(5,3)
# c = [[1,0,0],[0,1,0],[0,0,1]]

# pearsonr similarity
def pearsonr(vec1, vec2):
    pcc,p = stats.pearsonr(vec1, vec2)
    return pcc


# Eucliadean distance
def getSim(mat, sim='euclidean'):
    if cmp(sim, 'euclidean') == 0:
        return squareform(pdist(mat, 'euclidean'))
    elif cmp(sim, 'minkowski') == 0:
        return squareform(pdist(mat, 'minkowski', p=2))
    elif cmp(sim, 'cityblock') == 0 or cmp(sim, 'manhattan')== 0:
        return squareform(pdist(mat, 'cityblock'))
    elif cmp(sim, 'seuclidean') == 0:
        return squareform(pdist(mat, 'seuclidean', V=None))
    elif cmp(sim, 'sqeuclidean') == 0:
        return squareform(pdist(mat, 'sqeuclidean'))
    elif cmp(sim, 'cosine') == 0:
        return squareform(pdist(mat, 'cosine'))
    elif cmp(sim, 'correlation') == 0:
        return squareform(pdist(mat, 'correlation'))
    elif cmp(sim, 'hamming') == 0:
        return squareform(pdist(mat, 'hamming'))
    elif cmp(sim, 'jaccard') == 0 or cmp(sim, 'tanimoto') == 0:
        return squareform(pdist(mat, 'jaccard'))
    elif cmp(sim, 'chebyshev') == 0:
        return squareform(pdist(mat, 'chebyshev'))
    elif cmp(sim, 'canberra') == 0:
        return squareform(pdist(mat, 'canberra'))
    elif cmp(sim, 'braycurtis') == 0:
        return squareform(pdist(mat, 'braycurtis'))
    elif cmp(sim, 'mahalanobis') == 0:
        return squareform(pdist(mat, 'mahalanobis', VI=None))
    elif cmp(sim, 'sokalsneath') == 0:
        return squareform(pdist(mat, 'sokalsneath'))
    elif cmp(sim, 'pearsonr') == 0:
        return squareform(pdist(mat, lambda u, v: pearsonr(u,v)))
    else:
        print 'Error: cannot find similarity metrics, check your input.'
