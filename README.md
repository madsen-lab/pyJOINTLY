# JOINTLY : Multi-task clustering of single-cell RNA sequencing datasets
![Graphical Summary](figs/graphical_summary.png)

JOINTLY is an algorithm to jointly cluster single cell RNA sequencing datastes using kernel based joint non-negative matrix factorizaiton, allowing it to jointly non-linear manifolds. JOINTLY effectively captures shared cell states and while robustly accounting for dataset specific states.


# Installation

JOINTLY can be installed directly from GitHub or using pip. 

```{bash}
pip install git+https://github.com/madsen-lab/JOINTLY
pip install JOINTLY
```

# Usage

JOINTLY uses a list of scRNA-seq expression objects e.g. [scanpy](https://github.com/scverse/scanpy) or raw expression matrices as numpy arrays. 
Here we demonstrate how to make a JOINTLY object and how to run the clustering algorithm. 


```{py}
## Load libraries
import scanpy as sc
from jointly import jointly, CreateJointlyObject_from_scanpyList


## Load test data
data_list = [sc.read_h5ad('pancdata/panc1.h5ad'), 
             sc.read_h5ad('pancdata/panc2.h5ad'), 
             sc.read_h5ad('pancdata/panc3.h5ad')]


## Make JOINTLY object
jointlyObject = CreateJointlyObject_from_scanpyList(data_list)

## Running JOINTLY with default parameters
jointly(jointlyObject)

## Clustering matrix is availible through
jointlyObject.anndata.obsm['X_Jointly']

## Individual clustering matrices is availible through
ds = 0
jointlyObject.adata_list[ds].obsm['X_Jointly']

```


> _For any requests, please reach out to: <br/>[Jesper Madsen](jgsm@imada.sdu.dk) or [Andreas Møller](andreasfm@bmb.sdu.dk)_


### Citation
The preprint version of this article is available at [BioRxiv](https://doi.org/XXXXX)   <br/>
_Møller AF, Madsen JGS et al **Multi-task clustering of scRNA-seq datasets (2022)** Unpublished_  <br/>
