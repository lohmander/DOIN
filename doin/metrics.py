import torchmetrics as tm
import torch.nn.functional as F
from typing import List, Tuple
from torch import Tensor, nn


class DOINMetrics(nn.Module):
    def __init__(self, graph_thresholds=[0.25, 0.5, 0.75]):
        super().__init__()

        self.graph_thresholds = graph_thresholds

        self.retrieval_map = tm.RetrievalMAP()
        self.accuracy = tm.Accuracy(task="binary")
        self.f1 = tm.F1Score(task="binary")
        self.precision = tm.Precision(task="binary")
        self.recall = tm.Recall(task="binary")

    def forward(
        self,
        input_embeddings: Tensor,
        phrase_embeddings: Tensor,
        graph_input: Tensor,
        svos: List[List[Tuple[int, int, int]]],
    ):
        """
        Params:
            input_embeddings: (N, k, d)
            phrase_embeddings: (m, d)
            graph_input: (N, k, k, k)
            svos: List of lists of SVO triplets (one list per sample), essentially with shape (N, l, 3)
        """

        similarities = (
            F.normalize(input_embeddings, dim=-1)
            @ F.normalize(phrase_embeddings, dim=-1).T
        )
        graphs = graph_input.sigmoid()
        preds = {}
        threshold = min(self.graph_thresholds)

        for sim_i, graph_i, svos_i in zip(similarities, graphs, svos):
            preds_i = {}

            for edge in (graph_i >= threshold).nonzero():
                s, o, v = edge.tolist()
                svo = (
                    sim_i[s].argmax(-1).item(),
                    sim_i[v].argmax(-1).item(),
                    sim_i[o].argmax(-1).item(),
                )
                preds_i[svo] = max(preds_i.get(svo, 0), graph_i[s, o, v].item())

            for svo in set(list(preds_i.keys()) + svos_i):
                if svo not in preds:
                    preds[svo] = dict(true=[], pred=[])

                preds[svo]["true"].append(1.0 if svo in svos_i else 0.0)
                preds[svo]["pred"].append(preds_i.get(svo, 0.0))

        y_pred = []
        y_true = []
        y_idx = []

        for i, (_, y) in enumerate(preds.items()):
            y_pred.extend(y["pred"])
            y_true.extend(y["true"])
            y_idx.extend([i] * len(y["true"]))

        y_pred, y_true, y_idx = (
            torch.tensor(y_pred),
            torch.tensor(y_true).bool(),
            torch.tensor(y_idx),
        )

        ret_y_pred = y_pred[y_pred > 0]
        ret_y_true = y_true[y_pred > 0]
        ret_y_idx = y_idx[y_pred > 0]

        metrics = {}

        for threshold in self.graph_thresholds:
            mask = y_pred > threshold
            metrics.update(
                {
                    f"svo_{threshold}_accuracy": self.accuracy(
                        y_pred[mask], y_true[mask]
                    )
                    if mask.sum() > 0
                    else 0,
                    f"svo_{threshold}_f1": self.f1(y_pred[mask], y_true[mask])
                    if mask.sum() > 0
                    else 0,
                    f"svo_{threshold}_precision": self.precision(
                        y_pred[mask], y_true[mask]
                    )
                    if mask.sum() > 0
                    else 0,
                    f"svo_{threshold}_recall": self.recall(y_pred[mask], y_true[mask])
                    if mask.sum() > 0
                    else 0,
                }
            )

        metrics.update(
            {
                "svo_retrieval_map": self.retrieval_map(
                    ret_y_pred, ret_y_true, indexes=ret_y_idx
                )
                if len(ret_y_true) > 0
                else 0.0,
            }
        )

        return metrics
