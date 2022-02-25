import numpy as np
import pandas as pd
import scipy.sparse as sps
from hnmchallenge.dataset import Dataset
from hnmchallenge.recommender_interface import ItemSimilarityRecommender
from hnmchallenge.utils.sparse_matrix import truncate_top_k
from sklearn.metrics.pairwise import cosine_similarity
from sparsesvd import sparsesvd


class ItemKNN(ItemSimilarityRecommender):
    name = "ItemKNN"

    def __init__(self, dataset: Dataset, topk: int):
        super().__init__(dataset=dataset)
        self.topk = topk

    def compute_similarity_matrix(self, interaction_df: pd.DataFrame) -> None:
        sparse_interaction = self.dataset.get_user_item_interaction_matrix(
            interaction_df
        )
        sim = cosine_similarity(
            sparse_interaction.T, sparse_interaction.T, dense_output=True
        )
        # setting diag to 0 preventing considering in topk similarity self-similarities
        np.fill_diagonal(sim, 0)
        sim = truncate_top_k(sim, self.topk)
        # sim = sps.csr_matrix(sim)
        self.similarity_matrix = sim
