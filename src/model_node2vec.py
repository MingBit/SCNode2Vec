#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:52:31 2019

@author: mingwu
"""
import networkx as nx
import numpy as np
import gensim
import pandas as pd
import sklearn.metrics as sm
import inspect
from tqdm import tqdm, trange
from functools import reduce, partial
import multiprocessing as mp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


def _build_coexp_graph(ExpMatrix, method='pearson', return_graph=False):
    """
    build gene co-expression graph from raw gene expression matrix
    """

    if method == 'cosine':
        corrMatrix = pd.DataFrame(sm.pairwise.cosine_similarity(ExpMatrix), columns=ExpMatrix.index,
                                  index=ExpMatrix.index)
    else:
        corrMatrix = ExpMatrix.T.corr(method=method)

    corrMatrix[corrMatrix == 1] = 0
    # convert adjMatrix to edgelist
    List = list()
    for source in corrMatrix.index.values:
        for target in corrMatrix.index.values:
            if source != target:
                List.append((target, source, corrMatrix[source][target]))

    edgeList = pd.DataFrame(List, columns=['source', 'target', 'weight'])
    if return_graph:
        return nx.from_pandas_edgelist(edgeList, 'source', 'target', edge_attr='weight')
    else:
        return edgeList


def _biased_randomWalk(args):
    Expr, graph, start_node, walk_len, num_walks, p, q = args

    """
    biased random walk to generate multiple vectors from high-dimension graph
    Expr: gene expression matrix
    graph: gene co-expression network
    start_node: the starting point of the random walk
    walk_len: the number of walk steps
    num_walks: the repeat time of random walk 
    p: return parameter controls the likelihood of immediately revisiting a node in the walk
    q: in-out parameter allows to search to differentiate between 'inward' and 'outward' nodes
    """

    gene_var = Expr.mean(1)
    outer = tqdm(total=num_walks, desc='nodes', position=1)
    vec_nodes = list()
    for i in range(num_walks):
        vec_tmp = list()
        vec_len = 0
        vec_tmp.append(start_node)
        while vec_len < walk_len - 1:

            neighours = list(nx.neighbors(graph, vec_tmp[-1]))
            prob_list = list()

            for node in neighours:
                if len(vec_tmp) < 2:
                    alpha = 1 / q
                else:
                    alpha = 1 / p if node == vec_tmp[-2] else 1 if node in list(
                        nx.neighbors(graph, vec_tmp[-2])) else 1 / q

                prob = float(alpha) * abs(graph.get_edge_data(node, vec_tmp[-1])['weight']) * gene_var[node]
                prob_list.append(prob)

            #            next_node = neighours[prob_list.index(max(prob_list))]
            vec_tmp.append(np.random.choice(neighours, size=1, replace=False,
                                            p=[x / sum(prob_list) for x in prob_list])[0])
            vec_len = vec_len + 1
        outer.update(1)
        vec_nodes.append(vec_tmp)

    return vec_nodes


def _build_NN_Model(vector_list, size, **parameter_list) -> gensim.models.Word2Vec:
    """
    build neural network for graph embedding
    vector_list: a list of vectors generated from random walk
    size: number of dimensions after embedding
    """
    return gensim.models.Word2Vec(vector_list, size=size, **parameter_list)


def _binary_classifier(embedded_node, reference_links, param_grid, **kwargs):
    """
    build binary classifier for classification
    embedded_node: a data frame with rows corresponding to genes and columns representing embedded dimensions
    reference_links: adjacency edges links for training
    param_grid: inherited parameters from RandomForestClassifier
    """

    embedded_edges = pd.DataFrame(list(embedded_node.loc[reference_links.iloc[i]['source']]) +
                                  list(embedded_node.loc[reference_links.iloc[i]['target']])
                                  for i in range(reference_links.shape[0]))

    train_features, test_features, train_targets, test_targets = train_test_split(
        embedded_edges, reference_links['weight'].abs(),
        train_size=0.8,
        test_size=0.2,
        random_state=1,
        stratify=reference_links['weight'].abs())

    classifier = RandomForestClassifier(random_state=1, **kwargs)

    CV_rfc = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=10, n_jobs=10)
    CV_rfc.fit(train_features, train_targets)

    prediction_training_targets = CV_rfc.predict(train_features)
    self_accuracy = accuracy_score(train_targets, prediction_training_targets)
    print("Accuracy for training data (self accuracy):", self_accuracy)

    prediction_test_targets = CV_rfc.predict(test_features)
    test_accuracy = accuracy_score(test_targets, prediction_test_targets)
    print("Accuracy for test data:", test_accuracy)

    prediction_prob = CV_rfc.predict_proba(embedded_edges)
    prediction_type = CV_rfc.predict(embedded_edges)

    prediction_all_df = pd.DataFrame({'source': reference_links.source,
                                      'target': reference_links.target,
                                      'weight': [(1 - prediction_prob[i][0]) for i in range(len(prediction_type))]})
    return prediction_all_df


def run_node2vec(Expr, method, p, q, walk_len, num_walks, size,
                 workers, n_pc, **kwargs):
    """
    run pca and build node2vec model
    Expr: gene expression matrix
    method: correlation method to build gene co-expression network
    p: return parameter controls the likelihood of immediately revisiting a node in the walk
    q: in-out parameter allows to search to differentiate between 'inward' and 'outward' nodes
    walk_len: the number of walk steps
    num_walks: the repeat time of random walk
    size: number of dimensions after embedding
    workers: number of threads requested for parallel running
    n_pc: top n components selected from PCA
    """
    word2vec_args = [k for k, v in inspect.signature(gensim.models.Word2Vec).parameters.items()]
    word2vec_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in word2vec_args}

    # rf_classifier_args = [k for k, v in inspect.signature(RandomForestClassifier).parameters.items()]
    # rf_classifier_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in rf_classifier_args}

    print('-----running PCA------')
    pca = PCA(n_components=n_pc)
    pca.fit(Expr.T)
    top_pcs = pd.DataFrame(pca.components_).T

    print('-----running node2vec------')
    graph = _build_coexp_graph(Expr, method=method, return_graph=True)
    pool = mp.Pool(workers)
    num_of_nodes = len(graph.nodes)

    graph_list = [graph for _ in range(num_of_nodes)]
    walk_len_list = [walk_len for _ in range(num_of_nodes)]
    num_walks_list = [num_walks for _ in range(num_of_nodes)]
    p_list = [p for _ in range(num_of_nodes)]
    q_list = [q for _ in range(num_of_nodes)]
    Expr = [Expr for _ in range(num_of_nodes)]
    node_list = list(graph.nodes)

    vector_list = reduce(lambda x, y: x + y, pool.map(_biased_randomWalk,
                                                      zip(Expr, graph_list, node_list,
                                                          walk_len_list, num_walks_list,
                                                          p_list, q_list)))

    node2vec_model = _build_NN_Model(vector_list, dimensions=size, **word2vec_dict)
    node_vector = dict()
    for node in list(graph.nodes):
        node_vector[node] = node2vec_model.wv.get_vector(node)

    node_vector = pd.DataFrame.from_dict(node_vector).T
    top_pcs.index = node_vector.index
    merge_node_vector = pd.concat([node_vector, top_pcs], axis=1)

    # node2vec_prob = _binary_classifier(merge_node_vector, reference_links, param_grid, **rf_classifier_dict)
    pool.close()

    return merge_node_vector
