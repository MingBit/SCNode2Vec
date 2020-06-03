#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:16:47 2019

@author: mwu
"""
from __future__ import absolute_import
import os
import pandas as pd
import sys
import numpy as np
import itertools
import copy
from src import model_compare as compare
from src import model_node2vec as nv
import matplotlib.pyplot as plt
from arboreto.algo import grnboost2, genie3

sys.path.append(os.getenv('HOME') + '/MING_V9T/PhD_Pro/PySCNet')
from PySCNet.Preprocessing import gnetdata
import importlib

importlib.reload(compare)

if __name__ == '__main__':

    path = os.getenv('HOME') + '/MING_V9T/PhD_Pro/Test/Simulation/BoolODE_Data/'
    filename = sys.argv[1]
    p = sys.argv[2]
    q = sys.argv[3]
    dropRate = sys.argv[4]

    Sim, Sim_Ref, Sim_Ref_dropout, top_edges = compare.read_data(path + filename, float(dropRate))

    input_path = os.getenv("HOME") + '/MING_V9T/PhD_Pro/Test/Simulation/Node2Vec_BoolODE_Res/'
    output_path = os.getenv("HOME") + '/MING_V9T/PhD_Pro/Test/Simulation/Node2Vec_BoolODE_Res/PCA_ROC_Res_2/'

    gne_Sim_GENIE3 = genie3(np.asmatrix(Sim.T), gene_names=list(Sim.index))
    gne_Sim_GENIE3.columns = ['source', 'target', 'weight']
    gne_Sim_GRNBOOST2 = grnboost2(np.asmatrix(Sim.T), gene_names=list(Sim.index))
    gne_Sim_GRNBOOST2.columns = ['source', 'target', 'weight']

    param_grid = {
        'n_estimators': [50, 100, 500],
        'max_depth': [10, 20, 50],
        'max_features': [3, 5],
        'min_samples_leaf': [3, 4, 5],
        'criterion': ['gini', 'entropy']
    }

    node_matrix = nv.run_node2vec(Sim, Sim_Ref_dropout, method='cosine',
                                  p=1.0, q=0.1, size=5, walk_len=5,
                                  num_walks=1000, n_pc=5, workers=4)

    node_matrix_corr = nv._build_coexp_graph(node_matrix, method='cosine')
    TMP = node_matrix_corr.reindex(node_matrix_corr.weight.abs().sort_values(ascending=False).index)
    TMP.weight = np.concatenate((np.repeat(1, 140), np.repeat(0, 730)), axis=None)

    gne_Sim_node2vec = nv._binary_classifier(node_matrix, Sim_Ref_dropout, param_grid)

    # run PIDC
    Sim.to_csv('Expr.txt', sep='\t')
    os.system('julia run_pidc.jl')
    gne_Sim_PIDC = pd.read_csv('links.txt', sep='\t')
    gne_Sim_PIDC.columns = ['source', 'target', 'weight']

    PIDC_dict = dict(zip(['fpr', 'tpr', 'pre', 'recall_list', 'auc', 'avg_pre', 'precision', 'recall', 'f1_score'],
                         compare.test_run(links=gne_Sim_PIDC.reindex(
                             gne_Sim_PIDC.weight.abs().sort_values(ascending=False).index).head(
                             top_edges), Ref_links=Sim_Ref, input_dataset=Sim)))

    GENIE3_dict = dict(zip(['fpr', 'tpr', 'pre', 'recall_list', 'auc', 'avg_pre', 'precision', 'recall', 'f1_score'],
                           compare.test_run(links=gne_Sim_GENIE3.reindex(
                               gne_Sim_GENIE3.weight.abs().sort_values(ascending=False).index).head(
                               top_edges), Ref_links=Sim_Ref, input_dataset=Sim)))

    GRNBOOST2_dict = dict(zip(['fpr', 'tpr', 'pre', 'recall_list', 'auc', 'avg_pre', 'precision', 'recall', 'f1_score'],
                              compare.test_run(links=gne_Sim_GRNBOOST2.reindex(
                                  gne_Sim_GRNBOOST2.weight.abs().sort_values(ascending=False).index).head(
                                  top_edges), Ref_links=Sim_Ref, input_dataset=Sim)))

    Node2Vec_dict = dict(zip(['fpr', 'tpr', 'pre', 'recall_list', 'auc', 'avg_pre', 'precision', 'recall', 'f1_score'],
                             compare.test_run(links=gne_Sim_node2vec.reindex(
                                 gne_Sim_node2vec.weight.abs().sort_values(ascending=False).index).head(
                                 top_edges), Ref_links=Sim_Ref, input_dataset=Sim)))

    node_dict_list = list()
    node_dict_list.append(Node2Vec_dict)

    for i in range(10):
        gne_Sim_node2vec = gnetdata.load_Gnetdata_object(input_path +
                                                         filename + '_PCA/' + filename + '_node2vec_p_' +
                                                         str(p) + '_q_' + str(q) + '_dim_5_walk_5_dropRate_' +
                                                         str(dropRate) + '_repeat_' + str(i + 1) + '.pk')
        top_links = gne_Sim_node2vec.NetAttrs['links'].reindex(
            gne_Sim_node2vec.NetAttrs['links'].weight.abs().sort_values(ascending=False).index).head(top_edges)

        node_dict = dict(zip(['fpr', 'tpr', 'pre', 'recall_list', 'auc', 'avg_pre', 'precision', 'recall', 'f1_score'],
                             compare.test_run(links=top_links, Ref_links=Sim_Ref, input_dataset=Sim)))

        node_dict_list.append(node_dict)

    fig = plt.figure(figsize=(20, 15))
    grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0:])
    fig.suptitle(filename + '_p_' + str(p) + '_q_' + str(q) + ':Algorithms performance', fontsize=30)

    compare.build_curves(ax1, node_dict_list, GENIE3_dict, PIDC_dict, GRNBOOST2_dict, 'ROC', filename, p, q, dropRate)
    compare.build_curves(ax2, node_dict_list, GENIE3_dict, PIDC_dict, GRNBOOST2_dict, 'PCR', filename, p, q, dropRate)
    compare.build_plot(ax3, node_dict_list, GENIE3_dict, PIDC_dict, GRNBOOST2_dict)

    # fig.savefig(output_path + filename + '_PCA_p_' + str(p) + '_q_' + str(q) + '_dropRate_' +
    #             str(dropRate) + '_topTP.pdf', bbox_inches='tight')

    fig.savefig(os.getenv('HOME') + '/Desktop/test.pdf')

# p = 1.0
# q = 0.1
# dropRate = 0.75
# filename = 'G30_C1000_Clu_4_Tra_3-1000-1-50-0.5'

####################HSC data##########################
# path = os.getenv('HOME') + '/MING_V9T/PhD_Pro/TODOLIST/BN_Test/SC_Published_Data/'
# HSC_Data = pd.read_excel(path + 'HSC_DATA/Moignard_HSC_Data.xlsx', index_col=0, sheet_name=0).T
# HSC_Ref_raw = pd.read_csv(path + 'HSC_DATA/Moignard_HSC_ReferenceEdges.tsv', sep='\t')[['node1', 'node2']]
# HSC_Ref_raw.columns = ['source', 'target']
# HSC_Ref_raw['weight'] = 1
#
# Full_Ref = pd.DataFrame(itertools.permutations(HSC_Data.index, 2), columns=['source', 'target'])
# HSC_Ref = pd.merge(Full_Ref, HSC_Ref_raw, how='outer').fillna(0)
#
# HSC_Ref_dropout = copy.deepcopy(HSC_Ref)
# pos = list(HSC_Ref_dropout[abs(HSC_Ref_dropout.weight) == 1].index)
# replaceNum = np.random.choice(pos, int(len(pos) * dropRate), replace=False)
# HSC_Ref_dropout.loc[replaceNum, 'weight'] = 0
#
# Sim = HSC_Data
# Sim_Ref = HSC_Ref
# Sim_Ref_dropout = HSC_Ref_dropout
# top_edges = HSC_Ref_raw.shape[0]

#######################ESC data######################################
# ESC_Data = pd.read_excel(path + 'ESC_DATA/GSE59892_ESC_Dataset.xlsx', index_col=0, sheet_name=0)
# ESC_Gene_Symbol = pd.read_csv(path + 'ESC_DATA/GSE59892_ESC_Genesymobol.txt', sep='\t')
#
# ESC_Gene_Symbol_dict = dict(zip(ESC_Gene_Symbol.Ref_ID, ESC_Gene_Symbol.GeneSymbol))
# ESC_Ref_raw = pd.read_csv(path + 'ESC_DATA/GSE59892_ESC_ReferenceEdges.tsv', sep='\t')[['node1', 'node2']]
# ESC_Data.index = list(ESC_Gene_Symbol_dict.get(i) for i in list(ESC_Data.index))
# ESC_Data = ESC_Data[ESC_Data.sum(1) > 0]
#
# ESC_Ref_raw.columns = ['source', 'target']
# ESC_Ref_raw['weight'] = 1
#
# Full_Ref = pd.DataFrame(itertools.permutations(ESC_Data.index, 2), columns=['source', 'target'])
# ESC_Ref = pd.merge(Full_Ref, ESC_Ref_raw, how='outer').fillna(0)
#
# ESC_Ref_dropout = copy.deepcopy(ESC_Ref)
# pos = list(ESC_Ref_dropout[abs(ESC_Ref_dropout.weight) == 1].index)
# replaceNum = np.random.choice(pos, int(len(pos) * dropRate), replace=False)
# ESC_Ref_dropout.loc[replaceNum, 'weight'] = 0
#
# Sim = ESC_Data
# Sim_Ref = ESC_Ref
# Sim_Ref_dropout = ESC_Ref_dropout
# top_edges = ESC_Ref_raw.shape[0]


