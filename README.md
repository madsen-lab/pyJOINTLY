# JOINTLY

pyJOINTLY is an Python package for interpretable joint clustering of single-cell and single-nucleus RNA-seq, using kernel-based joint non-negative matrix factorization. JOINTLY effectively captures shared cell states, while robustly accounting for dataset-specific states. It works directly with [Scanpy](https://scanpy.readthedocs.io/en/stable/), a major single-cell genomics framework.

JOINTLY is also available as a R package [here](https://github.com/madsen-lab/rJOINTLY)

Scripts for reproducing the analyses in the manuscript are available [here](https://github.com/madsen-lab/JOINTLY_reproducibility). Please note, that all analyses for the manuscript was performed using the R version of JOINTLY. The R and the Python versions although very similar does not produce identical results. 

For the white adipose tissue atlas (WATLAS), the model weights are availiable [here](https://zenodo.org/deposit/8086433) and the atlas can be explored and downloaded [here](https://singlecell.broadinstitute.org/single_cell/study/SCP2289/an-integrated-single-cell-and-single-nucleus-rna-seq-white-adipose-tissue-atlas-watlas)

# Installation

JOINTLY can be installed directly from GitHub. 

```{bash}
pip install git+https://github.com/madsen-lab/pyJOINTLY
```

# Basic usage

JOINTLY uses a list of Scanpy objects or raw expression matrices as numpy arrays. 
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

# Citation
If you use JOINTLY in your work, please consider citing our manuscript:

_Møller AF, Madsen JGS et al. **Interpretable joint clustering of single-cell transcriptomes (2023)** Unpublished_  <br/>
