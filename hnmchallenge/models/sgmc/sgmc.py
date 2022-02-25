import numpy as np
import pandas as pd
import scipy.sparse as sps
from hnmchallenge.dataset import Dataset
from hnmchallenge.recommender_interface import ItemSimilarityRecommender
from sparsesvd import sparsesvd


class SGMC(ItemSimilarityRecommender):
    name = "SGMC"

    def __init__(self, dataset: Dataset, k: int = 256):
        super().__init__(dataset=dataset)
        self.k = k

    def compute_similarity_matrix(self, interaction_df: pd.DataFrame) -> None:
        sparse_interaction = self.dataset.get_user_item_interaction_matrix(
            interaction_df
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
        d_mat_i_inv = sps.diags(1 / d_inv)

        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsc()
        ut, s, vt = sparsesvd(norm_adj, self.k)
        D_U_U_T_D = d_mat_i @ vt.T @ vt @ d_mat_i_inv
        self.similarity_matrix = D_U_U_T_D
