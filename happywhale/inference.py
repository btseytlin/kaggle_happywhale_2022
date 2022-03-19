from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import CustomKNN


class InferenceModel:
    def __init__(self,
                 feature_extractor
                 ):
        self.feature_extractor = feature_extractor
        self.knn = CustomKNN(distance=CosineSimilarity())

        self.embeddings = None
        self.labels = None

    def fit(self, datamodule):
        train_embeddings = self.feature_extractor.get_embeddings(
            datamodule.train_dataloader(sampler=False, shuffle=False)
        )
        train_labels = datamodule.train.labels

        self.embeddings = train_embeddings
        self.labels = train_labels

    def _query_embeddings(self, query_embeddings, k):
        distances, indices = self.knn(
            query=query_embeddings,
            k=k,
            reference=self.embeddings,
        )

        result_labels = self.labels[indices]
        return distances, result_labels

    def query(self, query_dataloader, k=5):
        query_embeddings = self.feature_extractor.get_embeddings(query_dataloader)

        return self._query_embeddings(query_embeddings, k=k)
