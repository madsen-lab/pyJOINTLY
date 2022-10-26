

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
