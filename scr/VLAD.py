import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree

class VLAD:
  """
    Parameters
    ------------------------------------------------------------------
    k: int, default = 128
      Dimension of each visual words (vector length of each visual words)
    n_vocabs: int, default = 16
      Number of visual words
    Attributes
    ------------------------------------------------------------------
    vocabs: sklearn.cluster.Kmeans(k)
      The visual word coordinate system
    centers: [n_vocabs, k] array
      the centroid of each visual words
    db_VLADs: [no. images, flatten VLAD-vector]
      The vlad descriptors of all database images
  """
  def __init__(self, k=128, n_vocabs=16):
    self.n_vocabs = n_vocabs
    self.k = k
    self.vocabs = None
    self.centers = None
    self.db_VLADs = None
    self.tree = None

  def fit(self, X):
    """This function build a visual words dictionary and compute database VLADs,
    for a set of descriptor X.

    """
    X_matrix = np.vstack(X)
    self.vocabs = KMeans(n_clusters = self.n_vocabs, init='k-means++', tol=0.0001).fit(X_matrix)
    self.centers = self.vocabs.cluster_centers_
    self.db_VLADs = np.zeros([len(X), self.n_vocabs*self.k])
    for i, img_des in enumerate(X):
      v = self._calculate_VLAD(img_des)
      self.db_VLADs[i]=v
    self.tree = BallTree(self.db_VLADs) #query-tree, don't touch
    return self 
  
  def _calculate_VLAD(self, img_des):
    v = np.zeros([self.n_vocabs, self.k])
    NNs = self.vocabs.predict(img_des)
    for i in range(self.n_vocabs):
      if np.sum(NNs==i)>0:
        v[i] = np.sum(img_des[NNs==i, :]-self.centers[i], axis=0)
    v = v.flatten()
    v = np.sign(v)*np.sqrt(np.abs(v)) #power norm
    v = v/np.sqrt(np.dot(v,v))        #L2 norm
    return v

  def query(self, img_des, num_result=10):
    v = self._calculate_VLAD(img_des)
    _, idx = self.tree.query([v], num_result)
    return idx