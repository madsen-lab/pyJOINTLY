#Imports
from math import floor
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
from scipy.spatial import KDTree
import scanpy as sc
import graphtools
import ray
from tqdm.notebook import tqdm
from jointly.JointlyObject import JointlyObject
from fcmeans import FCM

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

import kneed



## Preprocess
def CreateJointlyObject_from_scanpy(adata, batch_key):
    """ """
    adata_list = []
    for batch in adata.obs[batch_key].unique():
        adata_list.append(adata[adata.obs[batch_key] == batch].copy())

    jointlyobject = JointlyObject(adata_list, batch_key = batch_key)
    for idx in range(len(adata_list)):
        if type(jointlyobject.adata_list[idx].X) is scipy.sparse.csr.csr_matrix:  #Check if matrix is sparse
            jointlyobject.adata_list[idx].X = np.asarray(jointlyobject.adata_list[idx].X .todense()) #Overwrite with dense array
        jointlyobject.X.append(jointlyobject.adata_list[idx].X)
    jointlyobject.anndata = adata
    jointlyobject.anndata.obs['Jointly_batch'] = jointlyobject.anndata.obs[batch_key]
    return jointlyobject

def CreateJointlyObject_from_scanpyList(adata_list, batch_key = None):
    """ """
    jointlyobject = JointlyObject(adata_list, batch_key = batch_key)
    for idx in range(len(adata_list)):
        if type(jointlyobject.adata_list[idx].X) is scipy.sparse.csr.csr_matrix:  #Check if matrix is sparse
            jointlyobject.adata_list[idx].X = np.asarray(jointlyobject.adata_list[idx].X .todense()) # overwrite with dense array
        jointlyobject.X.append(jointlyobject.adata_list[idx].X)

    jointlyobject.anndata = sc.AnnData.concatenate(*adata_list, batch_key = 'Jointly_batch') #use specific batch key, to not overwrite potential batch information
    return jointlyobject


## make method for numpy arrays



## CPCA
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
        jointlyobject.adata_list[ds].var.highly_variable = [x in hvgs[hvgs == True].index for x in jointlyobject.adata_list[ds].var.index]

def Normalize_libsize(jointlyobject, inplace = True, scalefactor = 10000, log = True):
    """Normalize to library size and subset to HVGs"""
    norm = []
    for i in range(jointlyobject.n_datasets):
        sf = jointlyobject.X[i].sum(axis = 1)[:,None] / scalefactor
        if log:
            norm.append(np.log1p((jointlyobject.X[i] / sf)[:,jointlyobject.adata_list[i].var['highly_variable']]))
            #norm.append(np.log1p(((jointlyobject.X[i] / jointlyobject.X[i].sum(axis = 1)[:,None]) * scalefactor)[:,jointlyobject.adata_list[i].var['highly_variable']]))
        else:
            norm.append((jointlyobject.X[i] / sf)[:,jointlyobject.adata_list[i].var['highly_variable']])
            #norm.append(((jointlyobject.X[i] / jointlyobject.X[i].sum(axis = 1)[:,None]) * scalefactor)[:,jointlyobject.adata_list[i].var['highly_variable']])


    if inplace:
        jointlyobject.norm_data =  norm
        print('Added normalized data to .norm_data')
    else:
        return norm

def Scale_data(jointlyobject, inplace = True):
    """Z-score genes and subset to HVGs """
    scaled = []
    scale = StandardScaler()
    for i in range(jointlyobject.n_datasets):
        scaled.append(scale.fit_transform(jointlyobject.norm_data[i].T).T)
    if inplace:
        jointlyobject.scaled_data =  scaled
        print('Added scaled data to .scaled_data')
    else:
        return scaled

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
            break
        if max(abs((d_new[idx] - d[idx]+ eps2) / (d[idx] + eps2))) < tol:  #added eps2 for division by zero error
            break
        d = d_new

    Q, _ = np.linalg.qr(np.dot(V, Q))
    B = np.dot(Q.T, V)   #instead of cross product
    sc_u, sc_s, sc_vh = np.linalg.svd(B)
    sc_u = np.dot(Q, sc_u)

    jointlyobject.common_components = {'u': sc_u, 's': sc_s, 'v':sc_vh}

    sc_u = sc_u[:, :k]
    sc_s = sc_s[:k]
    sc_vh = sc_vh[:k, :].T
    #sc_vh = sc_vh[:, :k]



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
                break
            if max(abs((d_new[idx] - d[idx]+ eps2) / (d[idx] + eps2))) < tol:  #added eps2 for division by zero error
                break
            d = d_new

        Q, _ = np.linalg.qr(np.dot(V, Q))
        B = np.dot(Q.T, V)   #instead of cross product
        si_u, si_s, si_vh = np.linalg.svd(B)
        si_u = np.dot(Q, si_u)

        si_u = si_u[:, :k]
        si_s = si_s[:k]
        #si_vh = si_vh[:, :k]
        si_vh = si_vh[:k, :].T
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
                break
            if max(abs((d_new[idx] - d[idx]+ eps2) / (d[idx] + eps2))) < tol:  #added eps2 for division by zero error
                break
            d = d_new

        Q, _ = np.linalg.qr(np.dot(V, Q))
        B = np.dot(Q.T, V)   #instead of cross product
        si_u, si_s, si_vh = np.linalg.svd(B)
        si_u = np.dot(Q, si_u)

        si_u = si_u[:, :k]
        si_s = si_s[:k]
        si_vh = si_vh[:k, :].T
        #si_vh = si_vh[:, :k]
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
            si_vh = si_vh[:k_sel, :].T
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






def make_kernel(jointlyobject, type = 'alphadecay', knn = 5, knn_max = 100, decay = 5, thresh = 1e-4, inplace = True):
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
                count2 = count1 / k + k - count1
                Snngraph[i][j]=count2
                Snngraph[j][i]=count2
    return Snngraph


@ray.remote
def SNN(pcs, k, prune = 1/15):

    neigh = NearestNeighbors(n_neighbors=k, radius=0, algorithm='kd_tree')
    neigh.fit(pcs)
    knn = neigh.kneighbors(pcs, n_neighbors=k, return_distance=False)
    knn = knn.astype(np.int)

    num_cells = knn.shape[0]
    rows = np.repeat(list(range(num_cells)), k)
    columns = knn.flatten()
    data = np.repeat(1, num_cells * k)
    snn = csr_matrix((data, (rows, columns)), shape=(num_cells, num_cells))

    snn = snn @ snn.transpose()

    rows, columns = snn.nonzero()
    data = snn.data / (k + (k - snn.data))
    data[data < prune] = 0

    return np.array(csr_matrix((data, (rows, columns)), shape=(num_cells, num_cells)).todense())

    #n_cells = pcs.shape[0]

    #dists = distance_matrix(pcs, pcs)
    #distanceMatrix=[[[(j,i), dists[j,i]] for i in range(n_cells)] for j in range(n_cells)]

    #similarityMatrix=findKNNList(distanceMatrix,k)
    #sharedNearestN= sharedNearest(n_cells,k)
    #sharedNearestN = np.array(sharedNearestN)
    #sharedNearestN[sharedNearestN < prune] = 0
    #return sharedNearestN


def Make_SNN(jointlyobject, neighbor_offset = 20, inplace = True, cpu = 1):
    """ """
    ray.init(num_cpus=cpu )
    SNNs = ray.get([SNN.remote(jointlyobject.adata_list[ds].obsm['X_pca'], #rice(jointlyobject.n_cells[ds]) +
                               neighbor_offset) for ds in range(jointlyobject.n_datasets)])

    ray.shutdown()

    if inplace:
        jointlyobject.SNN = SNNs
        print('Added SNNs to .SNN')
    else:
        return SNNs


def solve_for_W(X, H):
    W = np.linalg.lstsq(H.T, X.T, rcond = None)[0].T
    W[W < 0] = 0
    W = np.nan_to_num(W)
    return np.asarray(W)

def solve_for_F(K, H):
    F = np.linalg.lstsq(H.T, K, rcond = None)[0].T
    F[F < 0] = 0
    F = np.nan_to_num(F)
    return np.asarray(F)


def rice(n):
    return int(2*(n**(1/3)))

@ray.remote
def updateH(ds, data_range, rare, Fs, Ks, Hs, As, Ws, X, D_As,
                alpha, beta, lambda_, mu):
    js = list(data_range)
    js.remove(ds)

    numerator1 = np.multiply(np.dot(Fs[ds].T, Ks[ds]), rare[ds]*alpha)
    numerator2 = 2 * mu * Hs[ds]
    numerator3 = lambda_ * np.dot(Hs[ds], As[ds])
    numerator4 = sum([sum([beta  * np.linalg.multi_dot([Ws[j].T, X[ds]]),
                           beta  * np.linalg.multi_dot([Ws[ds].T, Ws[ds], Hs[ds]])])
                                  for j in js])

    denom1 = np.multiply(np.linalg.multi_dot([Fs[ds].T, Ks[ds], Fs[ds], Hs[ds]]), rare[ds]*alpha)

    denom2 = 2 * mu * np.linalg.multi_dot([Hs[ds], Hs[ds].T, Hs[ds]])
    denom3 = lambda_ * np.dot(Hs[ds], D_As[ds])
    denom4 = sum([sum([beta * np.linalg.multi_dot([Ws[j].T, Ws[j], Hs[ds]]),
                       2 * beta * np.linalg.multi_dot([Ws[ds].T, Ws[j], Hs[ds]]),
                       beta *  np.linalg.multi_dot([Ws[ds].T, X[ds]])])
                                          for j in js])
    return np.multiply(Hs[ds] , ((numerator1 + numerator2 + numerator3 + numerator4) / (denom1 + denom2 + denom3 + denom4)))




def JointlyDecomposition(jointlyobject, iter_max = 100, alpha = 100, mu = 1, lambda_ = 100, beta = 1, factorization_rank = 20, cpu = 1, emph_rare=True, initilization = 'NNDSVD', eps = 1e-20, early_stopping = True, cmeans_m = 5, scale_mode = 'standard_cells'):

    ray.init(num_cpus=cpu )

    n_genes = jointlyobject.norm_data[0].shape[1]#jointlyobject.adata_list[0].shape[1]
    #n_cells = [jointlyobject.adata_list[i].shape[0] for i in range(len(jointlyobject.adata_list))]

    scaler = MinMaxScaler()
    X = [scaler.fit_transform(x).T for x in jointlyobject.norm_data]

    #setting up NMF placeholders
    Fs = list()
    Hs = list()
    #Hs_new = list()
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
    #Clustering of all cPCAs at once
    #my_model = FCM(n_clusters=k)
    #my_model.fit(np.concatenate([jointlyobject.adata_list[i].obsm['X_pca'] for i in range(jointlyobject.n_datasets)]))
    #jointlyobject.anndata.obsm['F_clust'] = my_model.u
    #c_list = []
    #for i in jointlyobject.anndata.obs['Jointly_batch'].unique():
    #    c_list.append(jointlyobject.anndata[jointlyobject.anndata.obs['Jointly_batch'] == i].obsm['F_clust'])


    for ds in range(jointlyobject.n_datasets):

        if initilization == 'Clustering':
            #Clustering of all cPCAs at once
            #H_ = c_list[ds].T
            #Hs.append(c_list[ds].T)
            #Fs.append(solve_for_F(jointlyobject.K[ds], H_))

            #Individual lustering per dataset
            my_model = FCM(n_clusters=k, m = cmeans_m)
            my_model.fit(jointlyobject.adata_list[ds].obsm['X_pca'])
            H_ = my_model.u.T
            #initialize H
            Hs.append(H_)
            #initialize F
            Fs.append(solve_for_F(jointlyobject.K[ds], H_))


        else:
            #initialize H
            Hs.append(np.random.random_sample((k, jointlyobject.n_cells[ds])))
            #initialize F
            Fs.append(np.random.random_sample((jointlyobject.n_cells[ds], k)))

        #Parts of the Laplacian matrix
        Vs.append(np.sum(As[ds], axis = 0))
        D_As.append(np.diag(Vs[ds]))

        #rare cell
        if emph_rare:
            kr = rice(jointlyobject.n_cells[ds])
            dataA = pd.DataFrame(jointlyobject.adata_list[ds].obsm['X_pca']) # from
            dataB = pd.DataFrame(jointlyobject.adata_list[ds].obsm['X_pca']) # to

            kdB = KDTree(dataB.values)
            rare_neighbors = kdB.query(dataA.values, k=kr +1)[0]
            rare.append(np.array(1 - (1 / rare_neighbors[:,-1])))
        else:
            rare.append(np.ones((jointlyobject.n_cells[ds])))

    #Save the initializaiton
    jointlyobject.Hs_init = Hs

    #initialize W
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

    stop = False
    iterator = tqdm(range(iter_max))
    for _ in iterator:
        #Update H
        Hs_new = ray.get([updateH.remote(ds, data_range, rare, Fs_id, Ks_id, Hs_id, As_id, Ws_id, X_id, D_As_id,
            alpha, beta, lambda_, mu) for ds in range(jointlyobject.n_datasets)])

        Hs_new = [np.nan_to_num(x) for x in Hs_new]

        #Does not include most rescent score (stops before saving the updates)
        max_error = np.max([np.linalg.norm(Hs[ds] - Hs_new[ds]) for ds in range(jointlyobject.n_datasets)])
        if (max_error <= eps) and early_stopping == True:
            stop = True

        Hs = Hs_new


        Hs_id = ray.put(Hs)

        for ds in range(jointlyobject.n_datasets):
            #Update W
            Ws[ds] = solve_for_W(X[ds], Hs[ds])
            # Update F
            Fs[ds] = np.multiply(Fs[ds] , (np.dot(Ks[ds], Hs[ds].T) / np.linalg.multi_dot([Ks[ds], Fs[ds], Hs[ds], Hs[ds].T])))
        Ws_id = ray.put(Ws)
        Fs_id = ray.put(Fs)

        if stop == True:
            iterator.close()
            break

    ray.shutdown()
    Hs = [np.asarray(H) for H in Hs]
    Fs = [np.asarray(F) for F in Fs]
    jointlyobject.Hs = Hs
    jointlyobject.Fs = Fs
    jointlyobject.Ws = Ws

    scale = StandardScaler()
    for ds in range(jointlyobject.n_datasets):
        jointlyobject.adata_list[ds].obsm['X_Jointly'] = scale.fit_transform(scale.fit_transform(Hs[ds]).T)
        #scale_H(Hs[ds], mode = scale_mode)


    #merge_H = np.concatenate([scale_H(h, mode = scale_mode).T for h in Hs], axis = 1)
    #Scale all Hs together
    merge_H = np.concatenate([h for h in Hs], axis = 1)
    scale = StandardScaler()
    merge_H = scale.fit_transform(scale.fit_transform(merge_H).T)
    jointlyobject.anndata.obsm['X_Jointly'] = merge_H



def scale_H(H, mode = 'standard_cells'):
    if mode == 'standard_cells':
        scale = StandardScaler()
        return scale.fit_transform(H).T
    if mode == 'standard_features':
        scale = StandardScaler()
        return scale.fit_transform(H.T)
    if mode == 'standard_double':
        scale = StandardScaler()
        return scale.fit_transform(scale.fit_transform(H.T).T).T
    if mode == 'standard_double_r':
        scale = StandardScaler()
        return scale.fit_transform(scale.fit_transform(H).T)
    if mode == 'minmax_cells':
        scale = MinMaxScaler()
        return scale.fit_transform(H).T
    if mode == 'minmax_features':
        scale = MinMaxScaler()
        return scale.fit_transform(H.T)
    if mode == 'minmax_double':
        scale = MinMaxScaler()
        return scale.fit_transform(scale.fit_transform(H.T).T).T
    if mode == 'minmax_double_r':
        scale = MinMaxScaler()
        return scale.fit_transform(scale.fit_transform(H).T)

def GetModules(jointlyobject):
    scale = StandardScaler()
    ws = jointlyobject.Ws
    ws = [scale.fit_transform(scale.fit_transform(w.T).T) for w in ws]
    ws_sum = sum(ws) / len(ws)

    w = scale.fit_transform(scale.fit_transform(ws_sum.T).T)
    W = pd.DataFrame(w, index = jointlyobject.adata_list[1].var[jointlyobject.adata_list[1].var['highly_variable'] == True].index)
    modules = dict()
    for f in  W.columns:
        kneedle = kneed.KneeLocator(range(len(test.index)), W[f].sort_values(ascending = False),
                          S=2.0, curve="convex", direction="decreasing")
        modules[f] = list(W[f].sort_values(ascending = False)[:kneedle.knee].index)
    return modules


def jointly(jointlyobject, n_hvg_features = 1000, normalization_factor = 10000, log = False, scale = True,
            cPCA_threshold = 0.80, cPCA_kc = 20, cPCA_ki = 20, cPCA_oversampling = 10, cPCA_iter_max = 100,
            kernel_type = 'alphadecay', kernel_knn = 5, kernel_knn_max = 100, kernel_decay = 1, kernel_thresh = 1e-4,
            SNN_neighbor_offset = 20,
            decomposition_iter_max = 100, initilization = 'NNDSVD', decomposition_alpha = 100, decomposition_mu = 1, decomposition_lambda = 100, decomposition_beta = 1, decomposition_factorization_rank = 20, decomposition_emph_rare = True, decomposition_early_stopping = True,cmeans_m = 2,scale_mode = 'standard_cells',
            cpu = 1, return_adata = False):
    """
    Jointly main function

    TODO: add more more description here

    Attributes:
        jointlyobject (JointlyObject):
            JointlyObject constructed with:
                CreateJointlyObject_from_scanpy or
                CreateJointlyObject_from_scanpyList
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
        Normalize_libsize(jointlyobject, inplace = True, scalefactor = normalization_factor, log = log)
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
    JointlyDecomposition(jointlyobject, iter_max = decomposition_iter_max, alpha = decomposition_alpha, mu = decomposition_mu, lambda_ = decomposition_lambda, beta = decomposition_beta, factorization_rank = decomposition_factorization_rank, cpu = cpu, emph_rare = decomposition_emph_rare, initilization = initilization, early_stopping = decomposition_early_stopping,cmeans_m = cmeans_m, scale_mode = scale_mode)
    decomposition_params = decomposition_params | {'decomposition_iter_max' : decomposition_iter_max, 'decomposition_alpha' : decomposition_alpha, 'decomposition_mu' : decomposition_mu, 'decomposition_lambda' : decomposition_lambda, 'decomposition_beta' : decomposition_beta, 'decomposition_factorization_rank' : decomposition_factorization_rank}
    jointlyobject.parameters['Decomposition'] = decomposition_params
    if return_adata:
        return jointlyobject.anndata
