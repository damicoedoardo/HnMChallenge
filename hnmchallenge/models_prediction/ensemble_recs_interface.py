from hnmchallenge.models_prediction.recs_interface import RecsInterface
from hnmchallenge.stratified_dataset import StratifiedDataset


class EnsembleRecs(RecsInterface):
    RECS_NAME = None

    def __init__(
        self, models_list: list, kind: str, dataset: StratifiedDataset
    ) -> None:
        super().__init__(kind, dataset)
        # Check that all the models passed are extending Recs Interface
        assert len(
            models_list > 1
        ), "At least 2 models should be passed to create an Ensemble !"
        all(
            [issubclass(m, RecsInterface) for m in models_list]
        ), "Not all the models passed are extending `RecsInterface`"
        self.models_list = models_list

    def _create_ensemble_name(self):
        models_names = [model.RECS_NAME for model in self.models_list]
