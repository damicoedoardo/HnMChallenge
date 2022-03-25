import numpy as np
import pandas as pd
import scipy.sparse as sps
from hnmchallenge.dataset import Dataset
from hnmchallenge.recommender_interface import ItemSimilarityRecommender
from hnmchallenge.utils.sparse_matrix import interactions_to_sparse_matrix
from sparsesvd import sparsesvd


class SGMC(ItemSimilarityRecommender):
    name = "SGMC"

    def __init__(self, dataset: Dataset, k: int = 256, time_weight: bool = False):
        super().__init__(dataset=dataset, time_weight=time_weight)
        self.k = k

    def compute_similarity_matrix(self, interaction_df: pd.DataFrame) -> None:
        sparse_interaction, user_mapping_dict, _ = interactions_to_sparse_matrix(
            interaction_df,
            items_num=self.dataset._ARTICLES_NUM,
            users_num=None,
        )

        rowsum = np.array(sparse_interaction.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sps.diags(d_inv)
        norm_adj = d_mat.dot(sparse_interaction)

        colsum = np.array(sparse_interaction.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sps.diags(d_inv)
        d_mat_i = d_mat

        inv_d_inv = 1 / d_inv
        inv_d_inv[np.isinf(inv_d_inv)] = 0.0
        d_mat_i_inv = sps.diags(inv_d_inv)

        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsc()
        ut, s, vt = sparsesvd(norm_adj, self.k)
        D_U_U_T_D = d_mat_i @ vt.T @ vt @ d_mat_i_inv
        self.similarity_matrix = D_U_U_T_D
