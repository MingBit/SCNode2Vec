#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:16:47 2019
@author: mwu
"""

from __future__ import absolute_import
import pandas as pd
import numpy as np
import itertools
import copy
import seaborn as sns
from sklearn import metrics
from src import model_node2vec as nv

pd.set_option('mode.chained_assignment', None)


def read_data(file_path, dropRate=0):
    Sim = pd.read_csv(file_path + '/ExpressionData.csv', sep=',', index_col=0)
    Sim_Ref_raw = pd.read_csv(file_path + '/refNetwork.csv', sep=',', index_col=None)

    Sim_Ref_raw['weight'] = Sim_Ref_raw['Type'].map({'+': 1, '-': -1})
    Sim_Ref_raw = Sim_Ref_raw.drop(columns='Type')
    Sim_Ref_raw.columns = ['source', 'target', 'weight']

    Full_Ref = pd.DataFrame([f for f in itertools.product(Sim.index, repeat=2)], columns=['source', 'target'])
    Sim_Ref = pd.merge(Full_Ref, Sim_Ref_raw, how='outer').fillna(0)

    Sim_Ref_dropout = copy.deepcopy(Sim_Ref)
    pos = list(Sim_Ref_dropout[abs(Sim_Ref_dropout.weight) == 1].index)
    replaceNum = np.random.choice(pos, int(len(pos) * dropRate), replace=False)
    Sim_Ref_dropout.loc[replaceNum, 'weight'] = 0

    return [Sim, Sim_Ref, Sim_Ref_dropout, Sim_Ref_raw.shape[0]]


def repeat_node2vec(Sim, Sim_Ref, Sim_Ref_dropout, p, q, top_edges, param_grid, use_ref=False):

    node_matrix = nv.run_node2vec(Sim, p=p, q=q, size=10, walk_len=10,
                                  num_walks=1000, workers=10)
    if use_ref:
        gne_Sim_node2vec = nv._binary_classifier(embedded_node=node_matrix, reference_links=Sim_Ref_dropout,
                                                 select_n=0,
                                                 use_ref=True, param_grid=param_grid)
    else:
        gne_Sim_node2vec = nv._binary_classifier(embedded_node=node_matrix, reference_links=Sim_Ref_dropout,
                                                 select_n=int(Sim_Ref_dropout.shape[0] * 0.25),
                                                 param_grid=param_grid)
    Node2Vec_dict = dict(
        zip(['fpr', 'tpr', 'pre', 'recall_list', 'auc', 'avg_pre', 'precision', 'recall', 'f1_score'],
            test_run(links=gne_Sim_node2vec.reindex(
                gne_Sim_node2vec.weight.abs().sort_values(ascending=False).index).head(
                top_edges), Ref_links=Sim_Ref, input_dataset=Sim)))

    return Node2Vec_dict


def remove_duplicate(links):
    links_list = sorted(links[['source', 'target']].values.tolist())
    for i in range(len(links_list)):
        links_list[i] = tuple(sorted(links_list[i]))
    nodes = pd.DataFrame(list(set(links_list)), columns=('source', 'target'))
    links = pd.merge(links, nodes, how='right')

    return links


def mapping_edges(df_1, df_2, df_1_col_1, df_1_col_2, df_2_col_1, df_2_col_2):
    df_1['tmp1'] = df_1[df_1_col_1] + '_' + df_1[df_1_col_2]
    df_2['tmp1'] = df_2[df_2_col_1] + '_' + df_2[df_2_col_2]

    return len(set(df_1['tmp1']) & set(df_2['tmp1']))


def evaluation(links, Ref_links, Num_Genes):
    # links_filtered=links.loc[(abs(links['value']) > threshold) & (links['var1'] != links['var2'])]

    Detected = links.shape[0]
    Ref_links = Ref_links[Ref_links.weight != 0]
    TP = mapping_edges(links, Ref_links, 'source', 'target', 'source', 'target')
    FN = Ref_links.shape[0] - TP
    FP = Detected - TP
    TN = (Num_Genes * Num_Genes) - Ref_links.shape[0] - Detected + TP

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    FDR = FP / (TN + FP)

    F1_Score = (2 * Precision * Recall) / (Precision + Recall)
    print('Detected:', Detected)
    print('TP:', TP, '\n', 'FN:', FN, '\n', 'FP:', FP, '\n', 'TN:', TN)

    print('Precision:', Precision, '\n', 'Recall:', Recall, '\n',
          'FDR:', FP / (TN + FP))
    return Detected, TP, FN, FP, TN, Precision, Recall, FDR, F1_Score


def test_run(links, Ref_links, input_dataset, filename=None):
    Detected, TP, FN, FP, TN, Precision, Recall, FDR, F1_Score = evaluation(links, Ref_links, input_dataset.shape[0])

    Comp_Links = pd.merge(links, Ref_links, on=['source', 'target'], how='right').fillna(0)

    auc = metrics.roc_auc_score(np.array(Comp_Links['weight_y'].abs()), np.array(Comp_Links['weight_x'].abs()))
    fpr, tpr, threshold_1 = metrics.roc_curve(Comp_Links['weight_y'].abs(), Comp_Links['weight_x'].abs())
    pre, recall, threshold_2 = metrics.precision_recall_curve(Comp_Links['weight_y'].abs(),
                                                              Comp_Links['weight_x'].abs())
    # avg_pre = metrics.average_precision_score(Comp_Links['weight_y'].abs(), Comp_Links['weight_x'].abs())
    avg_pre_auc = metrics.auc(recall, pre)

    return [fpr, tpr, pre, recall, auc, avg_pre_auc, Precision, Recall, F1_Score]


def build_curves(ax, node_dict_list, GENIE3_dict, PIDC_dict, GRNBOOST2_dict,
                 curve, filename, p, q, dropRate):
    keywords = ['fpr', 'tpr'] if curve == 'ROC' else ['recall_list', 'pre']
    fill = 1 if curve == 'ROC' else 0

    node_fpr_pre_df = pd.DataFrame(node_dict_list[i][keywords[0]] for i in range(len(node_dict_list))).fillna(fill)
    node_tpr_recall_df = pd.DataFrame(node_dict_list[i][keywords[1]] for i in range(len(node_dict_list))).fillna(fill)
    node_auc_list = list(node_dict_list[i]['auc'] for i in range(len(node_dict_list)))
    node_avgpre_list = list(node_dict_list[i]['avg_pre'] for i in range(len(node_dict_list)))

    colors = sns.color_palette().as_hex() + sns.color_palette('hls', 8).as_hex()

    for i in range(len(node_dict_list)):
        ax.plot(node_dict_list[i][keywords[0]], node_dict_list[i][keywords[1]], color='grey', alpha=0.2)

    fpr_dict = {'GENIE3': GENIE3_dict[keywords[0]], 'PIDC': PIDC_dict[keywords[0]],
                'GRNBOOST2': GRNBOOST2_dict[keywords[0]],
                'Node2Vec': node_fpr_pre_df.iloc[0],
                'Node2Vec_Ref': node_fpr_pre_df.iloc[1]}

    tpr_dict = {'GENIE3': GENIE3_dict[keywords[1]], 'PIDC': PIDC_dict[keywords[1]],
                'GRNBOOST2': GRNBOOST2_dict[keywords[1]],
                'Node2Vec': node_tpr_recall_df.iloc[0],
                'Node2Vec_Ref': node_tpr_recall_df.iloc[1]}

    auc_dict = {'GENIE3': GENIE3_dict['auc'], 'PIDC': PIDC_dict['auc'], 'GRNBOOST2': GRNBOOST2_dict['auc'],
                'Node2Vec': node_auc_list[0], 'Node2Vec_Ref': node_auc_list[1]}

    avgpre_dict = {'GENIE3': GENIE3_dict['avg_pre'], 'PIDC': PIDC_dict['avg_pre'],
                   'GRNBOOST2': GRNBOOST2_dict['avg_pre'],
                   'Node2Vec': node_avgpre_list[0], 'Node2Vec_Ref': node_avgpre_list[1]}

    for i, color in zip(list(auc_dict.keys()), colors):
        ax.plot(fpr_dict[i], tpr_dict[i],
                label='ROC curve {0} (area = {1:0.2f})'.format(i, auc_dict[i]) if curve == 'ROC' else
                'PR curve {0} (area = {1:0.2f})'.format(i, avgpre_dict[i]),
                color=color)
    if curve == 'ROC':
        ax.plot([0, 1], [0, 1], 'k--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate' if curve == 'ROC' else 'Recall', fontsize=20)
    ax.set_ylabel('True Positive Rate' if curve == 'ROC' else 'Precision', fontsize=20)
    ax.set_title(filename + '_p_' + str(p) + '_q_' + str(q) + '_dropRate_' + str(dropRate), fontsize=12)
    ax.legend(loc="lower right")


def build_plot(ax, node_dict_list, GENIE3_dict, PIDC_dict, GRNBOOST2_dict):
    node_precision_list = list(node_dict_list[i]['precision'] for i in range(len(node_dict_list)))
    node_recall_list = list(node_dict_list[i]['recall'] for i in range(len(node_dict_list)))
    node_f1score_list = list(node_dict_list[i]['f1_score'] for i in range(len(node_dict_list)))

    Algorithms = ['GENIE3', 'PIDC', 'GRNBOOST2', 'Node2Vec', 'Node2Vec_Ref']
    Eva_Methods = ['Precision', 'Recall', 'F1_Score']

    data = [[GENIE3_dict['precision'], PIDC_dict['precision'], GRNBOOST2_dict['precision'],
             node_precision_list[0], node_precision_list[1]],
            [GENIE3_dict['recall'], PIDC_dict['recall'], GRNBOOST2_dict['recall'],
             node_recall_list[0], node_recall_list[1]],
            [GENIE3_dict['f1_score'], PIDC_dict['f1_score'], GRNBOOST2_dict['f1_score'],
             node_f1score_list[0], node_f1score_list[1]]]

    colors = sns.color_palette().as_hex() + sns.color_palette('hls', 8).as_hex()
    X = np.arange(5)
    for i in range(len(data)):
        print(data[i])
        ax.bar(X + 0.25 * i, data[i], color=colors[i], width=0.25)

    ax.set_xlabel('Algorithms', fontsize=25)
    ax.set_ylabel('Performance', fontsize=25)
    ax.set_xticks(X + 0.25)
    ax.set_xticklabels(Algorithms, fontsize=20)
    ax.legend(Eva_Methods, loc='upper left')
