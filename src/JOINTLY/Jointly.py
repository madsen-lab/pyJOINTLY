#Imports
from math import floor
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
from scipy.spatial import KDTree
import scanpy as sc
import graphtools
import ray
from tqdm.notebook import tqdm
from jointly.JointlyObject import JointlyObject


def CreateJointlyObject_from_scanpy(adata, batch_key):
    """ """
    adata_list = []
    for batch in adata.obs[batch_key].unique():
        adata_list.append(adata[adata.obs[batch_key] == batch].copy())
    jointlyobject = JointlyObject(adata_list, batch_key = batch_key)
    for data in adata_list:
        if type(data.X) is scipy.sparse.csr.csr_matrix:   # TODO: Check this works
            jointlyobject.X.append(data.X.todense())
        else:
            jointlyobject.X.append(data.X)

    jointlyobject.anndata = adata
    return jointlyobject
def CreateJointlyObject_from_scanpyList(adata_list, batch_key = None):
    """ """
    jointlyobject = JointlyObject(adata_list, batch_key = batch_key)
    for data in adata_list:
        if type(data.X) is scipy.sparse.csr.csr_matrix:   # TODO: Check this works
            jointlyobject.X.append(np.asarray(data.X.todense()))
        else:
            jointlyobject.X.append(data.X)

        #jointlyobject.adata_list.append(data)
    jointlyobject.anndata = sc.AnnData.concatenate(*adata_list, batch_key = 'Jointly_batch') #use specific batch key, to not overwrite potential batch information
    return jointlyobject
def CreateJointlyObject_from_numpyList(adata_list, batch_key = None):
    """ """
    pass #TODO: Finish method



def Normalize_libsize(jointlyobject, inplace = True, scalefactor = 10000):
    """Normalize to library size """
    norm = []
    for i in range(jointlyobject.n_datasets):
        norm.append(((jointlyobject.X[i] / jointlyobject.X[i].sum(axis = 1)[:,None]) * scalefactor)[:,jointlyobject.adata_list[i].var['highly_variable']])
    if inplace:
        jointlyobject.norm_data =  norm
        print('Added normalized data to .norm_data')
    else:
        return norm

def Scale_pearson_residuals(jointlyobject, inplace = True):
    """Normalize to pearson residuals. Can only be used as scaled data for cPCA but not for decomposition"""
    #https://scanpy-tutorials.readthedocs.io/en/latest/tutorial_pearson_residuals.html
    norm = []
    for i in range(jointlyobject.n_datasets):
        tmp = jointlyobject.adata_list[i].copy()
        sc.experimental.pp.normalize_pearson_residuals(tmp)
        norm.append(tmp.X[:,tmp.var['highly_variable']])
    if inplace:
        jointlyobject.scaled_data =  norm
        print('Added normalized data to .scaled_data')
    else:
        return norm


def Scale_data(jointlyobject, inplace = True):
    """Z-score genes and subset to HVGs """
    scaled = []
    scale = StandardScaler()
    for i in range(jointlyobject.n_datasets):
        scaled.append(scale.fit_transform(np.log1p(jointlyobject.norm_data[i].T)).T)
    if inplace:
        jointlyobject.scaled_data =  scaled
        print('Added scaled data to .scaled_data')
    else:
        return scaled

def Feature_selection(jointlyobject, n_hvg_features = 1000):
    """Feature selection using pearson residuals approximation"""
    for ds in range(len(jointlyobject.adata_list)):
        sc.experimental.pp.highly_variable_genes(
            jointlyobject.adata_list[ds], flavor="pearson_residuals", n_top_genes=n_hvg_features)
    Subset_to_HVG(jointlyobject)
    print("Added highly variable genes to .adata_list.var['highly_variable']")



def Subset_to_HVG(jointlyobject, HVGs = None):
    """ """
    hvg = pd.concat([jointlyobject.adata_list[ds].var.highly_variable for ds in range(jointlyobject.n_datasets)], axis = 1)
    hvg.columns = range(jointlyobject.n_datasets)
    for ds in range(jointlyobject.n_datasets):
        jointlyobject.adata_list[ds].var.loc[ jointlyobject.adata_list[ds].var.means== 0., 'highly_variable'] = np.NaN
    hvg = hvg.dropna()
    hvgs = hvg.sum(axis = 1) > 0
    for ds in range(jointlyobject.n_datasets):
        jointlyobject.adata_list[ds].var.highly_variable = [x in hvgs[hvgs == True].index for x in jointlyobject.adata_list[ds].var.index] #can probably be much faster


def cPCA(jointlyobject, threshold = 0.80, kc = 20, ki = 20, oversampling = 10, iter_max = 100):
    """Common PCA function"""

    X = jointlyobject.scaled_data
    n_datasets = len(X)
    #Calculate variance-covariance matrices
    V_ = []
    for i in range(n_datasets):
        V_.append(np.dot(X[i].T, X[i]) / (X[i].shape[0] -1))

    #Within group variance-covariance matrix
    V = np.zeros((X[i].shape[1], X[i].shape[1]))
    for i in range(n_datasets):
        V = V + V_[i] * X[1].shape[0] / sum([x.shape[0] for x in X])

    #Randomized SVD using QR decomposition on the V matrix
    k = kc

    eps2 = 2.220446e-16 ** (4/5)
    n = min(V.shape[1], k + oversampling)
    Q = np.random.random_sample((V.shape[1], n))
    d = np.zeros((k))
    tol=1e-5

    for iter_ in range(iter_max):
        Q, _ = np.linalg.qr(np.dot(V, Q))
        B = np.dot(V.T, Q)   #instead of cross product
        Q, _ = np.linalg.qr(B)
        _, s, _ = np.linalg.svd(B)
        d_new = s[:k]

        idx = d_new > eps2
        if sum(~idx) == 0:
            pass
        #    break
        if max(abs((d_new[idx] - d[idx]+ eps2) / (d[idx] + eps2))) < tol:  #added eps2 for division by zero error
            pass
        #    break
        d = d_new

    Q, _ = np.linalg.qr(np.dot(V, Q))
    B = np.dot(Q.T, V)   #instead of cross product
    sc_u, sc_s, sc_vh = np.linalg.svd(B)
    sc_u = np.dot(Q, sc_u)

    sc_u = sc_u[:, :k]
    sc_s = sc_s[:k]
    sc_vh = sc_vh[:, :k]
    sc_mprod = 2 * iter_ + 1

    var_explain = np.concatenate([np.var((np.dot(sc_u.T, X[ds].T)).T, axis = 0) / np.sum(np.var(X[ds],axis = 0)) for ds in range(n_datasets) ]).T


    var_explain_ind = pd.DataFrame(np.zeros((n_datasets, 3)))
    #Randomized SVD using QR decomposition on the V matrix
    for ds in range(n_datasets) :

        k = kc
        V = V_[ds]
        n = min(V.shape[1], k + oversampling)
        Q = np.random.random_sample((V.shape[1], n))
        d = np.zeros((k))
        tol=1e-5
        for iter_ in range(iter_max):
            Q, _ = np.linalg.qr(np.dot(V, Q))
            B = np.dot(V.T, Q)   #instead of cross product
            Q, _ = np.linalg.qr(B)
            _, s, _ = np.linalg.svd(B)
            d_new = s[:k]

            idx = d_new > eps2
            if sum(~idx) == 0:
                pass
            #    break
            if max(abs((d_new[idx] - d[idx]+ eps2) / (d[idx] + eps2))) < tol:  #added eps2 for division by zero error
                pass
            #    break
            d = d_new

        Q, _ = np.linalg.qr(np.dot(V, Q))
        B = np.dot(Q.T, V)   #instead of cross product
        si_u, si_s, si_vh = np.linalg.svd(B)
        si_u = np.dot(Q, si_u)

        si_u = si_u[:, :k]
        si_s = si_s[:k]
        si_vh = si_vh[:, :k]
        si_mprod = 2 * iter_ + 1

        #Save variances
        var_explain_ind.iloc[[ds],0] = int(ds)
        var_explain_ind.iloc[[ds],1] = np.sum(np.var((np.dot(sc_u.T, X[ds].T)).T, axis = 0) / np.sum(np.var(X[ds],axis = 0)).T)
        var_explain_ind.iloc[[ds],2] = np.sum(np.var((np.dot(si_u.T, X[ds].T)).T, axis = 0) / np.sum(np.var(X[ds],axis = 0)).T)


    ## Calculate residuals
    R_list = []
    for ds in range(n_datasets):
        R_list.append(V_[ds] - np.linalg.multi_dot([sc_u, sc_u.T, V_[ds]]))

    S_list = []
    for ds in range(n_datasets):
        ki = ki

        V = R_list[ds]
        n = min(V.shape[1], k + oversampling)
        Q = np.random.random_sample((V.shape[1], n))
        d = np.zeros((k))
        tol=1e-5

        for iter_ in range(iter_max):
            Q, _ = np.linalg.qr(np.dot(V, Q))
            B = np.dot(V.T, Q)   #instead of cross product
            Q, _ = np.linalg.qr(B)
            _, s, _ = np.linalg.svd(B)
            d_new = s[:k]

            idx = d_new > eps2
            if sum(~idx) == 0:
                pass
            #    break
            if max(abs((d_new[idx] - d[idx]+ eps2) / (d[idx] + eps2))) < tol:  #added eps2 for division by zero error
                pass
            #    break
            d = d_new

        Q, _ = np.linalg.qr(np.dot(V, Q))
        B = np.dot(Q.T, V)   #instead of cross product
        si_u, si_s, si_vh = np.linalg.svd(B)
        si_u = np.dot(Q, si_u)

        si_u = si_u[:, :k]
        si_s = si_s[:k]
        si_vh = si_vh[:, :k]
        si_mprod = 2 * iter_ + 1


        # Evaluate number of components
        var_exp = []
        for k_sel in range(ki):
            var_exp.append(np.sum(np.var(np.concatenate([np.dot(sc_u.T, X[ds].T).T, np.dot(si_u[:,:k_sel].T, X[ds].T).T], axis = 1), axis = 0)) / np.sum(np.var(X[ds],axis = 0)))

        k_sel = np.argmin(abs(np.array(var_exp) - float(var_explain_ind[var_explain_ind[0] == ds][2]) * threshold))

        if k_sel > 0:
            si_u = si_u[:, :k_sel]
            si_s = si_s[:k_sel]
            si_vh = si_vh[:, :k_sel]
            S_list.append({'u':si_u,
                           's':si_s,
                           'vh':si_vh})
    ## Final PC space
    common = sc_u
    if len(S_list) != 0:
        for i in S_list:

            individual = i['u']
            common = np.concatenate([common, individual], axis = 1)


    C_list = []
    for ds in range(n_datasets):
        C_list.append(np.dot(common.T, X[ds].T).T)

    for ds in range(n_datasets):
        jointlyobject.adata_list[ds].obsm['X_pca'] = C_list[ds]
    print("Added cPCA to .adata_list.obsm['X_pca']")



def make_kernel(jointlyobject, type = 'alphadecay', knn = 5, knn_max = 100, decay = 2, thresh = 1e-4, inplace = True):
    """ """
    K = []
    if type == 'alphadecay':
        for i in range(jointlyobject.n_datasets):
            graph = graphtools.Graph(jointlyobject.adata_list[i].obsm['X_pca'],
                        n_pca=None,
                        n_landmark=None,
                        knn=knn,
                        knn_max=knn_max,
                        decay=decay,  # TODO: What happens when alpha = 1
                        thresh=thresh)
            K.append((graph.build_kernel_to_data(jointlyobject.adata_list[i].obsm['X_pca']) + graph.build_kernel_to_data(jointlyobject.adata_list[i].obsm['X_pca']).T).todense() / 2 )  #Check if I have to transpose PCA from anndata
    else:  #TODO: Make error if not alphadecay
        pass
    if inplace:
        jointlyobject.K = K
        print('Added {} kernels to .K'.format(type))
    else:
        return K


def findKNNList(distanceMatrix, k):
    count=len(distanceMatrix)
    global similarityMatrix #TODO: Get rid of the global
    count=len(distanceMatrix)
    similarityMatrix = [[0 for i in range(k)] for j in range(count)]
    for i in range(count):
        matrixsorted = sorted(distanceMatrix[i], key=lambda x: x[1])
        for j in range(k):
            similarityMatrix[i][j] = int(matrixsorted[j+1][0][1])
    return similarityMatrix

def countIntersection(listi,listj):
    intersection=0
    for i in listi:
        if i in listj:
            intersection=intersection+1
    return intersection

def sharedNearest(count,k):
    Snngraph= [[0 for i in range(count)] for j in range(count)]
    for i in range(0,count-1):
        nextIndex=i+1
        for j in range(nextIndex,count):
            if j in similarityMatrix[i] and i in similarityMatrix[j]:
                count1=countIntersection(similarityMatrix[i],similarityMatrix[j])
                Snngraph[i][j]=count1
                Snngraph[j][i]=count1
    return Snngraph


@ray.remote
def SNN(pcs, k):
    n_cells = pcs.shape[0]

    dists = distance_matrix(pcs, pcs)
    distanceMatrix=[[[(j,i), dists[j,i]] for i in range(n_cells)] for j in range(n_cells)]

    similarityMatrix=findKNNList(distanceMatrix,k)
    sharedNearestN= sharedNearest(n_cells,k)
    sharedNearestN = np.array(sharedNearestN)
    return sharedNearestN


def Make_SNN(jointlyobject, neighbor_offset = 20, inplace = True, cpu = 1):
    """ """
    ray.init(num_cpus=cpu )
    SNNs = ray.get([SNN.remote(jointlyobject.adata_list[ds].obsm['X_pca'], rice(jointlyobject.n_cells[ds]) + neighbor_offset) for ds in range(jointlyobject.n_datasets)])

    ray.shutdown()

    if inplace:
        jointlyobject.SNN = SNNs
        print('Added SNNs to .SNN')
    else:
        return SNNs


def solve_for_W(X, H):
    W = np.linalg.lstsq(H.T, X.T, rcond = None)[0].T
    W[W < 0] = 0
    return W


def rice(n):
    return int(2*(n**(1/3)))

@ray.remote
def updateH(ds, data_range, rare, Fs, Ks, Hs, As, Ws, X, D_As,
                alpha, beta, lambda_, mu):
    js = list(data_range)
    js.remove(ds)

    numerator1 = np.multiply(rare[ds]*alpha,np.dot(Fs[ds].T, Ks[ds]))
    numerator2 = 2 * mu * Hs[ds]
    numerator3 = lambda_ * np.dot(Hs[ds], As[ds])
    numerator4 = sum([sum([beta  * np.linalg.multi_dot([Ws[j].T, X[ds]]),
                                       beta  * np.linalg.multi_dot([Ws[ds].T, Ws[ds], Hs[ds]])])
                                  for j in js])
    denom1 = np.multiply(rare[ds]*alpha , np.linalg.multi_dot([Fs[ds].T, Ks[ds], Fs[ds], Hs[ds]])) + 2*mu*np.linalg.multi_dot([Hs[ds], Hs[ds].T, Hs[ds]]) + lambda_* np.dot(Hs[ds], D_As[ds])
    denom2 = sum([sum([beta * np.linalg.multi_dot([Ws[j].T, Ws[j], Hs[ds]]),
                                   2*beta * np.linalg.multi_dot([Ws[ds].T, Ws[j], Hs[ds]]),
                                   beta *  np.linalg.multi_dot([Ws[ds].T, X[ds]])])
                                          for j in js])
    return np.multiply(Hs[ds] , ((numerator1 + numerator2 + numerator3 + numerator4) / (denom1 + denom2)))




def JointlyDecomposition(jointlyobject, iter_max = 100, alpha = 100, mu = 1, lambda_ = 100, beta = 1, factorization_rank = 20, cpu = 1):

    ray.init(num_cpus=cpu )

    n_genes = jointlyobject.adata_list[0].shape[1]
    n_cells = [jointlyobject.adata_list[i].shape[0] for i in range(len(jointlyobject.adata_list))]

    X = [x.T for x in jointlyobject.norm_data]

    #setting up NMF placeholders
    Fs = list()
    Hs = list()
    Hs_new = list()
    #As = list()
    Vs = list()
    D_As = list()
    rare = list()
    Ws = list()
    As = jointlyobject.SNN   #Replace throughout
    Ks = jointlyobject.K

    nns = 20
    k = factorization_rank   # Replace all the way through

    rare = []

    for ds in range(jointlyobject.n_datasets):
        Hs.append(np.random.rand(k, n_cells[ds]))
        Hs_new.append(np.random.rand(k, n_cells[ds]))
        Fs.append(np.random.rand(n_cells[ds], k))

        #Parts of the Laplacian matrix
        Vs.append(np.sum(As[ds], axis = 0))
        D_As.append(np.diag(Vs[ds]))

        #rare cell
        kr = rice(n_cells[ds])
        dataA = pd.DataFrame(jointlyobject.adata_list[ds].obsm['X_pca']) # from
        dataB = pd.DataFrame(jointlyobject.adata_list[ds].obsm['X_pca']) # to

        kdB = KDTree(dataB.values)
        rare_neighbors = kdB.query(dataA.values, k=kr +1)[0]
        rare.append(np.array(1 - (1 / rare_neighbors[:,-1])))


    for ds in range(jointlyobject.n_datasets):

        Ws.append(solve_for_W(X[ds], Hs[ds]))

    data_range = range(jointlyobject.n_datasets)

    # Put variables into ray shared memory
    Fs_id= ray.put(Fs)
    Ks_id= ray.put(Ks)
    Hs_id= ray.put(Hs)
    As_id= ray.put(As)
    Ws_id= ray.put(Ws)
    X_id= ray.put(X)
    D_As_id= ray.put(D_As)

    for _ in tqdm(range(iter_max)):
        #Update H
        Hs = ray.get([updateH.remote(ds, data_range, rare, Fs_id, Ks_id, Hs_id, As_id, Ws_id, X_id, D_As_id,
            alpha, beta, lambda_, mu) for ds in range(jointlyobject.n_datasets)])

        Hs_id = ray.put(Hs)

        for ds in range(jointlyobject.n_datasets):
            Ws[ds] = solve_for_W(X[ds], Hs[ds])
            # Update F
            Fs[ds] = np.multiply(Fs[ds] , (np.dot(Ks[ds], Hs[ds].T) / np.linalg.multi_dot([Ks[ds], Fs[ds], Hs[ds], Hs[ds].T])))
        Ws_id = ray.put(Ws)
        Fs_id = ray.put(Fs)

    ray.shutdown()
    Hs = [np.asarray(H) for H in Hs]
    jointlyobject.Hs = Hs
    jointlyobject.Ws = Ws

    scale = StandardScaler()
    for ds in range(jointlyobject.n_datasets):
        jointlyobject.adata_list[ds].obsm['X_Jointly'] = scale.fit_transform(Hs[ds].T)


    merge_H = np.concatenate([scale.fit_transform(h.T).T for h in Hs], axis = 1)
    jointlyobject.anndata.obsm['X_Jointly'] = merge_H.T




def jointly(jointlyobject, n_hvg_features = 1000, normalization_factor = 10000, scale = True,
            cPCA_threshold = 0.80, cPCA_kc = 20, cPCA_ki = 20, cPCA_oversampling = 10, cPCA_iter_max = 100,
            kernel_type = 'alphadecay', kernel_knn = 5, kernel_knn_max = 100, kernel_decay = 1, kernel_thresh = 1e-4,
            SNN_neighbor_offset = 20,
            decomposition_iter_max = 100, decomposition_alpha = 100, decomposition_mu = 1, decomposition_lambda = 100, decomposition_beta = 1, decomposition_factorization_rank = 20,
            cpu = 1, return_adata = False):
    """
    Jointly main function

    TODO: add more more description here

    Attributes:
        jointlyobject (JointlyObject):
            JointlyObject constructed with:
                CreateJointlyObject_from_scanpy,
                CreateJointlyObject_from_scanpyList or
                CreateJointlyObject_from_numpyList
        n_hvg_features (int):
            Number of highly variable genes to select for per dataset pearson residuals.
            HVGs are overlapped over datasets only using genes expressed in all datsets.
            If 'None' - User defined HVG selection is required and stored in jointlyobject.adata_list[dataset].var['highly_variable'].
            Default: 1000
        normalization_factor (int, float or None):
            Option to library size normalize all cells to a common factor.
            If 'None' - User defined normalization is required to be present in jointlyobject.adata_list[dataset].X (all values must be positive)
            Default: 10000
        scale (bool):
            Option to scale the datasets to mean 0 and unit variance
            Default: True
        cPCA_threshold (float):
            Default: 0.80
        cPCA_kc (int):
            Default: 20
        cPCA_ki (int):
            Default: 20
        cPCA_oversampling (int):
            Default: 10
        cPCA_iter_max (int):
            Default: 100
        kernel_type (str):
            Options: ('alphadecay', 'User')
            If 'User' - User defined kernels is required to be present in jointlyobject.K
            Default: 'alphadecay'
        kernel_knn (int):
            Default: 5
        kernel_knn_max (int):
            Default: 100
        kernel_decay (int):
            Default: 2
        kernel_thresh (float):
            Default: 1e-4
        SNN_neighbor_offset (int):
            Default: 20
        decomposition_iter_max (int):
            Default: 100
        decomposition_alpha (int):
            Default: 100
        decomposition_mu (int):
            Default: 1
        decomposition_lambda (int):
            Default: 100
        decomposition_beta (int):
            Default: 1
        decomposition_factorization_rank (int):
            Default: 20
        cpu (int):
            Default: 1



    """
    #Optional HVG selection
    if n_hvg_features is None:

        jointlyobject.parameters['Preprocessing']['n_hvg_features'] = 'User feature selection'
        pass    #TODO: Implement user HVG selection
    elif type(n_hvg_features) is int:
        Feature_selection(jointlyobject, n_hvg_features = n_hvg_features)
        jointlyobject.parameters['Preprocessing']['n_hvg_features'] = n_hvg_features
    else:
        raise TypeError('n_hvg_features must be an integer or None')

    #Optional normalizaiton
    if normalization_factor is None:
        jointlyobject.parameters['Preprocessing']['normalization'] = 'User normalizaiton'
        pass    #TODO: Implement user normalizaiton
    elif (type(normalization_factor) is int) or (type(normalization_factor) is float):
        Normalize_libsize(jointlyobject, inplace = True, scalefactor = normalization_factor)
        jointlyobject.parameters['Preprocessing']['normalization'] = normalization_factor
    else:
        raise TypeError('normalization_factor must be an integer or float')

    #Optional data scaling
    if scale is False:
        jointlyobject.parameters['Preprocessing']['scaling'] = 'User scaling'
        pass #TODO: Implement user scale
    elif scale is True:
        Scale_data(jointlyobject)
        jointlyobject.parameters['Preprocessing']['scaling'] = 'Z-score'
    else:
        raise TypeError('scale must be a bool')

    #TODO: Implement checks
    cPCA(jointlyobject, threshold = cPCA_threshold, kc = cPCA_kc, ki = cPCA_ki, oversampling = cPCA_oversampling, iter_max = cPCA_iter_max)
    jointlyobject.parameters['Dimmensionality_reduction'] = {'Reduction type': 'cPCA', 'threshold' :cPCA_threshold, 'kc' : cPCA_kc, 'ki' : cPCA_ki, 'oversampling' : cPCA_oversampling, 'iter_max' : cPCA_iter_max}

    #TODO: Implement checks
    make_kernel(jointlyobject, type = kernel_type, knn = kernel_knn, knn_max = kernel_knn_max, decay = kernel_decay, thresh = kernel_thresh, inplace = True)
    decomposition_params = {'kernel_type' : kernel_type, 'kernel_knn' : 'kernel_kernel_knn', 'kernel_knn_max' : kernel_knn_max, 'kernel_decay' : kernel_decay, 'kernel_thresh' : kernel_thresh}

    #TODO: Implement checks
    Make_SNN(jointlyobject, neighbor_offset = SNN_neighbor_offset, inplace = True, cpu = cpu)
    decomposition_params['SNN_neighbor_offset' ] = SNN_neighbor_offset

    #TODO: Implement checks
    JointlyDecomposition(jointlyobject, iter_max = decomposition_iter_max, alpha = decomposition_alpha, mu = decomposition_mu, lambda_ = decomposition_lambda, beta = decomposition_beta, factorization_rank = decomposition_factorization_rank, cpu = cpu)
    decomposition_params = decomposition_params | {'decomposition_iter_max' : decomposition_iter_max, 'decomposition_alpha' : decomposition_alpha, 'decomposition_mu' : decomposition_mu, 'decomposition_lambda' : decomposition_lambda, 'decomposition_beta' : decomposition_beta, 'decomposition_factorization_rank' : decomposition_factorization_rank}
    jointlyobject.parameters['Decomposition'] = decomposition_params
    if return_adata:
        return jointlyobject.anndata
