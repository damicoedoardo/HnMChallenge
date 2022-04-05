import numpy as np
import pandas as pd
import scipy.sparse as sps
from hnmchallenge.recommender_interface import ItemSimilarityRecommender
from hnmchallenge.utils.sparse_matrix import interactions_to_sparse_matrix
from scipy.sparse.linalg import eigsh
from sparsesvd import sparsesvd


class PSGE(ItemSimilarityRecommender):
    name = "PSGE"

    def __init__(
        self,
        dataset,
        k: int = 10,
        alpha: float = 0.5,
        time_weight: bool = False,
    ):
        super().__init__(dataset=dataset, time_weight=time_weight)
        self.k = k
        self.alpha = alpha

    def compute_similarity_matrix(self, interaction_df: pd.DataFrame) -> None:
        sp_int, user_mapping_dict, _ = interactions_to_sparse_matrix(
            interaction_df,
            items_num=self.dataset._ARTICLES_NUM,
            users_num=None,
        )

        # computing user mat
        user_degree = np.array(sp_int.sum(axis=1))
        d_user_inv = np.power(user_degree, -self.alpha).flatten()
        d_user_inv[np.isinf(d_user_inv)] = 0.0
        d_user_inv_diag = sps.diags(d_user_inv)

        d_user = np.power(user_degree, self.alpha).flatten()
        d_user[np.isinf(d_user)] = 0.0
        d_user_diag = sps.diags(d_user)

        item_degree = np.array(sp_int.sum(axis=0))
        d_item_inv = np.power(item_degree, -self.alpha).flatten()
        d_item_inv[np.isinf(d_item_inv)] = 0.0
        d_item_inv_diag = sps.diags(d_item_inv)

        d_item = np.power(item_degree, self.alpha).flatten()
        d_item[np.isinf(d_item)] = 0.0
        d_item_diag = sps.diags(d_item)

        int_norm = d_user_inv_diag.dot(sp_int).dot(d_item_inv_diag)
        gram_matrix = int_norm.T @ int_norm

        # compute eigendecomposition of the gram matrix
        print("Computing eigendecomposition can take time...")
        eigenvalues, eigenvectors = eigsh(gram_matrix, k=self.k, which="LA")
        print("Done!")
        sim = (d_item_diag @ eigenvectors * eigenvalues**2) @ eigenvectors.T
        self.similarity_matrix = sim
