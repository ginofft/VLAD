from pathlib import Path
from typing import Dict, List, Union, Optional
from ImageDataset import ImageDataset
from sklearn.cluster import KMeans
import numpy as np
import h5py
from utils import *
from Descriptors import compute_SIFT

conf={
   'SIFT': {
        'output': 'feats-SIFT',
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
            'resize_force': False,
        },
    },
}

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
  """
  def __init__(self, k=128, n_vocabs=16):
    self.n_vocabs = n_vocabs
    self.k = k
    self.vocabs = None
    self.centers = None
    self.tree = None

  def fit(self,
          conf,
          img_dir:Path, 
          out_path: Optional[Path] = None,
          overwrite:bool = False):
    """This function build a visual words dictionary and compute database VLADs,
    and export them into a h5 file in 'out_path'

    Args
    ----------------------------------------------------------------------------
    conf: local descripors configuration
    img_dir: database image directory
    out_path: 
    """
    #Setup dataset and output path
    dataset = ImageDataset(img_dir,conf)
    if out_path is None:
      out_path = Path(img_dir, conf['output']+'.h5')
    out_path.parent.mkdir(exist_ok=True, parents=True)

    features = [data['feature'] for data in dataset] 
    X = np.vstack(features)
    self.vocabs = KMeans(n_clusters = self.n_vocabs, init='k-means++').fit(X)
    self.centers = self.vocabs.cluster_centers_

    db_VLADs = np.zeros([len(dataset), self.n_vocabs*self.k])
    for i,data in enumerate(dataset):
      name = dataset.names[i]
      v = self._calculate_VLAD(data['feature'])
      with h5py.File(str(out_path), 'a', libver='latest') as fd:
        try:
          if name in fd:
            del fd[name]
          grp = fd.create_group(name)
          grp.create_dataset('vlad', data=v)
          grp.create_dataset('feature', data=features[i])
        except OSError as error:
          if 'No space left on device' in error.args[0]:
            del grp, fd[name]
          raise error
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

  def query(self,
        query_dir: Path,
        vlad_features: Path, 
        out_path: Optional[Path] = None,
        n_result=10):
    #define output path
    if out_path is None:
      out_path = Path(query_dir, 'retrievals'+'.h5')
    out_path.parent.mkdir(exist_ok=True, parents=True)

    query_names = [str(ref.relative_to(query_dir)) for ref in query_dir.iterdir()]
    images = [read_image(query_dir/r) for r in query_names]
    query_vlads = np.zeros([len(images), self.n_vocabs*self.k])
    for i, img in enumerate(images):
      query_vlads[i] = self._calculate_VLAD(compute_SIFT(img))

    #TODO
    with h5py.File(str(vlad_features), 'r', libver = 'latest') as f:
      db_names = []
      db_vlads = np.zeros([len(f.keys()), self.n_vocabs*self.k])
      for i, key in enumerate(f.keys()):
        data = f[key]
        db_names.append(key)
        db_vlads[i]= data['vlad'][()]

    sim = np.einsum('id, jd -> ij', query_vlads, db_vlads)
    pairs = pairs_from_similarity_matrix(sim, n_result)
    pairs = [(query_names[i], db_names[j]) for i,j in pairs]
    retrieved_dict = {}

    for query_name, db_name in pairs:
      if query_name in retrieved_dict.keys():
        retrieved_dict[query_name].append(db_name)
      else:
        retrieved_dict[query_name] = [db_name]

    with h5py.File(str(out_path), 'a', libver='latest') as f:
      try:
        for k,v in retrieved_dict.items():
          if k in f:
            del f[k]
          f[k] =v
      except OSError as error:
        if 'No space left on device' in error.args[0]:
          pass
        raise error
    return self