# -*- coding: cp936 -*-
from __future__ import division
import sys, re, csv
import numpy as np
import pandas as pd
import networkx as nx
import community
from sklearn.cluster import spectral_clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn import mixture
from sim import getSim
import scripts.script_module as mod

# load functional matrix produced by Gene Ontology
def loadMat(fn):
    fi = open(fn)
    header = fi.readline()
    if 'wiphi' in fn:
        proteins = [s.strip('"') for s in header.upper().strip().split(',')]
        mat = np.array([map(float, line.strip().split(',')) for line in fi])
    else:
        proteins = [s.strip('"') for s in header.upper().strip().split('\t')]
        mat = np.array([map(float, line.strip().split('\t')) for line in fi])
    return proteins, mat

# load topological matrix produced by Graph Embedding
def loadArray(fn):
    data = pd.read_csv(fn,delim_whitespace=True,skiprows=1,header=None,names=range(0,64))
    # sort node index by num-order
    data = data.sort_index()
    numlist = list(data.index)
    matlist = data.as_matrix()
    return numlist, matlist

# load a ppi network, return a graph
def loadNetwork(fn):
    g = nx.Graph()
    for line in open(fn):
        strs = line.upper().strip().split("\t")
        g.add_edge(strs[0].strip(), strs[1].strip())
    return g

# get adjacency matrix of a graph 
def getAdj(g):
    length = g.number_of_nodes()
    nodelist = [str(num) for num in range(0,length)]
    # sort node index by num-order
    nodelist = sorted(g.node, key=lambda x: int(x))
    adj = nx.to_numpy_matrix(g, nodelist=nodelist)
    # delete the self-circle
    adj = adj - np.diag(adj)
    return adj

# get community of a graph by best_partition algorithm
def getCommunity(g):
    '''
    It uses the louvain method described in Fast unfolding of communities in large networks, 
    Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte, Renaud Lefebvre, 
    Journal of Statistical Mechanics: Theory and Experiment 2008(10), P10008 (12pp)
    '''
    ids = []
    # compute the best partition
    partition = community.best_partition(g)
    # size = float(len(set(partition.values())))
    for com in set(partition.values()):
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        ids.append(list_nodes)
    return ids
# mix topological matrix (produced by Graph Embedding) and functional matrix (produced by Gene Ontology)
def mixmat(func_mat, func_list, topo_mat, topo_list, metadata_filename, beta):
    index_nodeid_map = {}
    with open(metadata_filename,'rb') as metadata:
        reader = csv.DictReader(metadata)
        for row in reader:
            index_nodeid_map[int(row["node_id"])] = row["name"]
    # print index_nodeid_map
    new_func_mat = np.zeros_like(topo_mat)
    [x,y] = new_func_mat.shape
    print x,y
    func = []
    for i in range(0, x):
        for j in range(0, y):
            indic_topo_i = int(topo_list[i])
            protein1 = index_nodeid_map[indic_topo_i]
            indic_topo_j = int(topo_list[j])
            protein2 = index_nodeid_map[indic_topo_j]
            if protein1 in func_list and protein2 in func_list:
                indic_func_i = func_list.index(protein1)
                indic_func_j = func_list.index(protein2)
                func.append(func_mat[indic_func_i][indic_func_j])
            else:
                func.append(0.01)
    new_func_mat = np.asarray(func).reshape((x,y))
    # print new_func_mat
    return beta * new_func_mat + (1-beta) * topo_mat

# load index_nodeid_map from metadata of 'node_id <-> name'
def loadNodeIdMap(metadata_filename):
    index_nodeid_map = {}
    with open(metadata_filename,'rb') as metadata:
        reader = csv.DictReader(metadata)
        for row in reader:
            index_nodeid_map[int(row["node_id"])] = row["name"]
    return index_nodeid_map

# load index_node_map from metadata 'name <-> node_id'
def loadNodeMap(metadata_filename):
    index_node_map = {}
    with open(metadata_filename,'rb') as metadata:
        reader = csv.DictReader(metadata)
        for row in reader:
            index_node_map[row["name"]] = int(row["node_id"])
    return index_node_map

# get the mixing submatrix
def get_sub_mix_mat(func_mat, func_list, topo_mat, topo_list, index_node_map, dynamicPIN, gama):
    topo_mat = np.asarray(topo_mat)
    func_mat = np.asarray(func_mat)
    sub_mat_length = len(dynamicPIN)
    sub_topo_mat = np.zeros((sub_mat_length,sub_mat_length))
    sub_func_mat = np.zeros((sub_mat_length,sub_mat_length))

    for i in range(0,sub_mat_length):
        indic_topo_i = index_node_map[dynamicPIN[i]] if dynamicPIN[i] in index_node_map else 'no'
        indic_func_i = func_list.index(dynamicPIN[i]) if dynamicPIN[i] in func_list else 'no'
        for j in range(0,sub_mat_length):
            indic_topo_j = index_node_map[dynamicPIN[j]] if dynamicPIN[j] in index_node_map else 'no'
            indic_func_j = func_list.index(dynamicPIN[j]) if dynamicPIN[j] in func_list else 'no'
            if indic_topo_i != 'no' and indic_topo_j != 'no':
                sub_topo_mat[i,j] = topo_mat[indic_topo_i,indic_topo_j]
            else:
                sub_topo_mat[i,j] = 0.01
            if indic_func_i != 'no' and indic_func_j != 'no':
                sub_func_mat[i,j] = func_mat[indic_func_i,indic_func_j]
            else:
                sub_func_mat[i,j] = 0.01
    index_sub_mix_mat = []
    for item in dynamicPIN:
        index_sub_mix_mat.append(index_node_map[item])
    sub_mix_mat = gama*sub_func_mat + (1-gama)*sub_topo_mat
    return index_sub_mix_mat, sub_mix_mat

# get dynamicPINs
def get_dynamicPINs(ppiService, funcService, func_lvl = 2):
    d_func_genes = funcService.get_d_func_genes(func_lvl)
    dynamicPINs = []
    for func_name in d_func_genes:
        dpin = ppiService.get_dynamicPIN(d_func_genes[func_name])
        if len(list(dpin)) <= 1:
            continue
        dynamicPINs.append(list(dpin))
    return dynamicPINs

# map clusters_id to proteins_name
def maps(clusters_id, metadata_filename):
    index_nodeid_map = {}
    clusters_name=[]
    with open(metadata_filename,'rb') as metadata:
        reader = csv.DictReader(metadata)
        for row in reader:
            index_nodeid_map[int(row["node_id"])] = row["name"]
    for cluster_id in clusters_id:
        if cluster_id == '':
            break
        node_ids = [int(index) for index in cluster_id if index]
        names = [index_nodeid_map[nid] for nid in node_ids]
        clusters_name.append(names)
    return clusters_name

# AgglomerativeClustering
def hclustering(matlist, numlist, k):
    ids = []
    hi = AgglomerativeClustering(n_clusters=k, affinity="l2", linkage='complete').fit_predict(np.asarray(matlist))
    labels = hi
    for i in range(0,k):
        list_id = np.asarray(numlist)[labels==i]
        ids.append(list(list_id))
    return ids

# mixture.GaussianMixture
def gclustering(matlist, numlist, k):
    ids = []
    gmm = mixture.GaussianMixture(n_components=k, random_state=0).fit(np.asarray(matlist))
    labels = gmm.predict(np.asarray(matlist))
    # proba = gmm.predict_proba(np.asarray(matlist))
    # print proba, len(proba)
    # for line in proba:
    #     print line, len(line)
    for i in range(0,k):
        list_id = np.asarray(numlist)[labels==i]
        ids.append(list(list_id))
    return ids

# mixture.BayesianGaussianMixture
def dpgclustering(matlist, numlist, k):
    ids = []
    gmm = mixture.BayesianGaussianMixture(n_components=k, weight_concentration_prior_type='dirichlet_process').fit(np.asarray(matlist))
    labels = gmm.predict(np.asarray(matlist))
    # proba = gmm.predict_proba(np.asarray(matlist))
    # print proba, len(proba)
    # for line in proba:
    #     print line, len(line)
    for i in range(0,k):
        list_id = np.asarray(numlist)[labels==i]
        ids.append(list(list_id))
    return ids

# KMeans
def kclustering(matlist, numlist, k):
    ids = []
    kmeans = KMeans(n_clusters=k, random_state=0).fit(np.asarray(matlist))
    labels = kmeans.labels_
    for i in range(0,k):
        list_id = np.asarray(numlist)[labels==i]
        ids.append(list(list_id))
    return ids
# DBSCAN
def dbscan(matlist, numlist, eps):
    ids = []
    db = DBSCAN(eps=eps, min_samples=5).fit(np.asarray(matlist))
    labels = db.labels_
    k = len(set(labels)) - (1 if -1 in labels else 0) # k is the numbers of clusters
    for i in range(0,k):
        list_id = np.asarray(numlist)[labels==i]
        ids.append(list(list_id))
    return ids
# Birch
def bclustering(matlist, numlist, thre):
    ids = []
    brc = Birch(branching_factor=80, n_clusters=300, threshold=thre, compute_labels=True)
    brc.fit(np.asarray(matlist,dtype=float))
    brc.predict(np.asarray(matlist,dtype=float))
    labels = brc.labels_
    k = len(set(labels)) 
    # k is the numbers of clusters, including the bad cluster label, for instance -1
    for i in range(0,k):
        list_id = np.asarray(numlist)[labels==i]
        ids.append(list(list_id))
    return ids

# spectral clustering
def spclustering(matlist, numlist, k):
    ids = []
    labels = spectral_clustering(matlist, n_clusters=k)
    k = len(set(labels)) 
    # k is the numbers of clusters, including the bad cluster label, for instance -1
    for i in range(0,k):
        list_id = np.asarray(numlist)[labels==i]
        ids.append(list(list_id))
    return ids

# original clustering algorithm by Bin Xu
def clustering(mat, k, names, size=2):
    labels = spectral_clustering(mat, n_clusters=k)
    clusters = dict()
    for a, clu_id in enumerate(labels):
        clusters.setdefault(clu_id, set())
        clusters[clu_id].add(a)
    name_clusters = list()
    for c_id in clusters:
        cluster = clusters[c_id]
        name_cluster = [names[c] for c in cluster]
        if len(name_cluster) < size:
            continue
        name_clusters.append(name_cluster)
    return name_clusters

# detect protein complex
def detecting(g, clusters, alpha):
    cands = list()
    for cluster in clusters:
        # print cluster
        subgraph = g.subgraph(cluster)
        # print subgraph
        for conn in nx.connected_component_subgraphs(subgraph):
            if len(conn)<2:
                continue
            
            # all neighs of conn
            neighs = set()
            for c in conn:
                for v in g[c]:
                    neighs.add(v)
            cand = set(conn.nodes())
            # check link
            for neigh in neighs:
                links = 0
                for c in conn:
                    if g.has_edge(neigh, c): links += 1
                if float(links)/len(conn) > alpha:
                    cand.add(neigh)
            cands.append(cand)
    return cands

# merge overlapping protein complex
def merging(preds, g, thre_ov=0.8):
    def f_den(pred, g):
        ne = len(g.subgraph(pred).edges())
        nv = len(pred)
        return ne / (nv * (nv - 1) / 2)

    def f_ov(pred1, pred2):
        common = len(set.intersection(set(pred1), set(pred2)))
        return float(common * common) / (len(pred1) * len(pred2))
        #return common * common / len(pred1) * len(pred2)

    ov_g = nx.Graph()
    for i in range(len(preds) - 1):
        for j in range(i + 1, len(preds) - 1):
            ov = f_ov(preds[i], preds[j])
            # print 'ov', ov
            if ov >= 0.8:
                ov_g.add_edge(i, j)

    new_preds = list()

    for i in range(len(preds)):
        if i not in ov_g:
            new_preds.append(preds[i])

    conns = nx.connected_components(ov_g)
    for conn in conns:
        # print 'conn', conn
        if len(conn) == 1:
            print '--'
            new_preds.append(preds[conn])
        else:
            # dens = [f_den(preds[p], g) for p in conn]
            # i = np.argmax(dens)
            # new_preds.append(preds[list(conn)[i]])
            temp = set()
            # print 'conn', conn
            for co in list(conn):
                # print 'preds[co]', preds[co]
                # print 'co', co
                temp = temp | preds[co]
            # print 'temp', temp
            new_preds.append(list(temp))
    return new_preds

# output the prediction
def output(cands, fn_out):
    fo = open(fn_out, 'w')
    for cand in cands:
        fo.write('\t'.join(cand) + '\n')
    fo.flush()
    fo.close()

def test():
    g_test = nx.Graph()
    g_test.add_edge('1','2')
    g_test.add_edge('1','3')
    g_test.add_edge('2','1')
    print g_test
    print g_test.node
    print g_test.number_of_edges()
    print g_test.number_of_nodes()
    adj = nx.to_numpy_matrix(g_test)
    print np.sum(adj==1)


if __name__ == '__main__':

    fn_ppi = 'data/graph/unweighted/%s.str.tab'  # ppi data
    # fn_ppi = 'data/ppi/%s.ppi.tab'  # ppi data
    fn_meta = 'data/meta/%s.meta.csv' # ppi's index and name
    # fn_emd = 'data/emb/unweighted/%s.emd.tab'    # ppi's representation in a low-dimension
    fn_emd = 'data/emb/pq/p=1andq=4/%s.emd.tab'    # ppi's representation in a low-dimension
    fn_func = 'data/wang/%s.wang.tab'
    fn_out = 'output/%s/%s.%d.%d.tab' # output

    # ppis = ['gavin', 'collins', 'krogan', 'wiphi']
    ppis = ['collins']


    # load funcat data
    fn_funcat = 'data/funcat-2.1_data_20070316'
    funcService = mod.FuncService(fn_funcat)

    # ppis = ['gavin']

    # parameter
    func_lvl = 2
    alphas = np.arange(10, 11)
    # betas = np.arange(2,40,2)
    thre_ov = 0.8

    for ppi in ppis:
        print ppi
        _fn_ppi = fn_ppi % (ppi,)
        _fn_meta = fn_meta % (ppi,)
        _fn_emd = fn_emd % (ppi,)
        _fn_func = fn_func % (ppi,)

        # get the numlist:[index of the protein] and the matlist [the feature vectors of the proteins]

        # load the graph from  PIN file
        g = loadNetwork(_fn_ppi)

        # load the index_node_map from metadata file
        index_node_map = loadNodeMap(_fn_meta)

        # Call the ppiService from PIN file
        ppiService = mod.PPIService(_fn_ppi)

        # load the functional file and calculate the topological similarity throghout 'minkowski' distance
        mat_name_f, mat_sim_f = loadMat(_fn_func)
        mat_num_t, matlist = loadArray(_fn_emd)
        mat_sim_t = getSim(matlist, 'minkowski')
        mat_sim_t = np.exp(-0.3*mat_sim_t/ mat_sim_t.std())

        dynamicPINs = get_dynamicPINs(ppiService, funcService, func_lvl)
        for gama in np.arange(2,3,1):
            clusters_name = []
            for dynamicPIN in dynamicPINs:
                index_sub_mix_mat, sub_mix_mat = get_sub_mix_mat(mat_sim_f,mat_name_f,mat_sim_t,mat_num_t,index_node_map,dynamicPIN,float(gama)/10)
                # print 'sub_mix_mat', sub_mix_mat
                cluster_id = kclustering(sub_mix_mat, index_sub_mix_mat, int(np.math.ceil(len(dynamicPIN) / 10)))
                for line in maps(cluster_id, _fn_meta):
                    if len(line) >= 2:
                        clusters_name.append(line)
            for alpha in alphas:
                cands = detecting(g, clusters_name, float(alpha) / 10)
                print 'before merge: ', len(cands)
                cands = merging(cands, g, thre_ov)
                print 'after merge: ', len(cands)
                _fn_out = fn_out % (ppi, ppi, alpha, gama)
                output(cands, _fn_out)
    print 'END'