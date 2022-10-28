import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
class JointlyObject:
    """
    Container to store Jointly compatible data

    TODO: add more more description here

    Attributes:
        adata_list (list):
            List of AnnData objects to jointly cluster (cells by genes)
        H (array):
            Cell clustering matrix (one matrix per dataset, dimensions cells by k)
        W (array):
            Dataset-specific gene loading factors (one matrix per dataset, dimensions k by genes)
        adata_joined (AnnData):
            Joined adata object with joint clustering and embedding
        parameters (dict):
            Dictionary of parameters used on this object
    """

    def __init__(self, adata_list, batch_key = None):
        self.parameters = {'Preprocessing':{},
                           'Dimmensionality_reduction':{},
                           'Decomposition':{}}
        self.anndata = None
        self.adata_list = adata_list
        self.batch_key = batch_key
        self.X = []
        self.n_datasets = len(adata_list)
        self.n_cells = [self.adata_list[i].shape[0] for i in range(self.n_datasets)]


    def __repr__(self):
        reply = "Jointly object with {} datasets\n".format(self.n_datasets )
        #TODO: Change prompt based on availible attributes
        #if hasattr(self, 'batch_key'):
        #    self.batch_key
        reply = reply + 'Total cells: {}'.format(sum(self.n_cells))
        return reply


    def ExportParams(self):
        pass


    def ComputeEmbedding(self):
        """Make a joint embedding of the merged dataset"""
        sc.pp.neighbors(self.anndata , n_neighbors = 20, metric = 'cosine', key_added = 'Jointly_neighbors', use_rep = 'X_Jointly' )
        sc.tl.umap(self.anndata, neighbors_key = 'Jointly_neighbors')
        self.anndata.obsm['UMAP_Jointly'] = self.anndata.obsm['X_umap']


    def ComputeEmbeddingIndividual(self):
        """Make a embeddings per dataset"""
        #TODO: make tests to see of ComputeEmbedding has been run
        for ds in range(self.n_datasets):
            sc.pp.neighbors(self.adata_list[ds] , n_neighbors = 20, metric = 'cosine', key_added = 'Jointly_neighbors', use_rep = 'X_Jointly' )
            sc.tl.umap(self.adata_list[ds], neighbors_key = 'Jointly_neighbors')
            self.adata_list[ds].obsm['UMAP_Jointly'] = self.adata_list[ds].obsm['X_umap']

    def PlotSharedEmbedding(self, cell_key = 'celltype', batch_key = 'Jointly_batch', cmap_celltype = 'tab20', cmap_batch = sc.pl.palettes.default_20, save_key = None):
        """Plot shared embeddings """
        #TODO: make tests to see of ComputeEmbedding has been run
        sc.pl.embedding(self.anndata, basis = 'UMAP_Jointly', color = cell_key, size = 30, add_outline=True, frameon = False, palette = cmap_celltype, save = save_key+'_'+'Celltype.pdf' if type(save_key) is str else False)
        sc.pl.embedding(self.anndata, basis = 'UMAP_Jointly', color = batch_key,  size = 20, add_outline=True,  frameon = False, palette = cmap_batch, save = save_key+'_'+'Batch.pdf' if type(save_key) is str else False)

    def PlotSharedEmbedding_split(self, batch_key, celltype_key, size=60, frameon=False, legend_loc=None):
        """Plot split view of shared embedding """
        #TODO: make tests to see of ComputeEmbedding has been run
        fig, ax = plt.subplots(1, self.n_datasets, figsize=(5*self.n_datasets, 5))
        for idx, batch in enumerate(list(set(self.anndata.obs[batch_key]))):
            tmp = self.anndata.copy()
            tmp.obs.loc[tmp.obs[batch_key] != batch, celltype_key] = np.NaN
            tmp1 = tmp[tmp.obs[batch_key] != batch]
            tmp2 = tmp[tmp.obs[batch_key] == batch]
            sc.pl.embedding(tmp1, basis = 'UMAP_Jointly', color = celltype_key, size = size*2, add_outline=True, frameon = frameon,ax = ax[idx], show = False, legend_loc = None, title = batch)
            sc.pl.embedding(tmp2, basis = 'UMAP_Jointly', color = celltype_key, size = size, add_outline=True, frameon = frameon, ax = ax[idx], show = False, title = batch, legend_loc = None if self.n_datasets != idx+1 else 'right margin')
