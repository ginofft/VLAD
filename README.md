# Vector of Locally Aggregated Descriptors
## TL;DR
Python implementation of VLAD using scikit-learn's Kmeans and opencv's SIFT. References:

H. Jégou, F. Perronnin, M. Douze, J. Sánchez, P. Pérez and C. Schmid, "Aggregating Local Image Descriptors into Compact Codes," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, no. 9, pp. 1704-1716, Sept. 2012, doi: 10.1109/TPAMI.2011.235.

## Dataset, Query
The current data consist of 795 images, taken from uncalibrated tourist's camera at 8 locations
- british museum entrance
- florence cathedral side
- lincoln memorial statue
- milan cathedral front
- mount rushmore
- piazza san marcro side
- sagrada familia front
- st pauls cathedral front
## Installation
```
pip install VLAD
```

## How to use
The demo code can be found in demo.ipynb, these are:

Setting up database, query and output directories.
```python
from pathlib import Path
database = Path('data') #folder storing database images
query_dir = Path('query') #folder storing query images
#output paths
output = Path('output') #folder in which our result would be stored
vlad_feature = output / 'vlads.h5' #storing all database's VLADs
retrieval = output / 'retrieval.h5' #storing query's retrieval results
```

Training VLAD for a new database.
```python
from VLAD.vlad import VLAD
vlad = VLAD() 
vlad.fit(database, vlad_feature) 
```

You can also load a trained vocab list (not database vlads).
```python
vocab_path = 'output/vocabs.joblib'
vlad.load(vocab_path)
```

Return queries.
```python
vlad.query(query_dir, vlad_feature, retrieval, n_result=40) 
```

Plotting
```python
from VLAD.utils import *
plot_retrievals_images(retrieval, query_dir, database)
```
