import torch
import torch.nn.functional as F
from typing import List, Tuple
from torch import Tensor, nn
from torchmetrics import (
    AUROC,
    AveragePrecision,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
)
from doin.matcher import Matcher


class DOINMetrics(nn.Module, Matcher):
    def __init__(self):
        super().__init__()

        self.rmap = RetrievalMAP()
        self.rmrr = RetrievalMRR()
        self.rdcg = RetrievalNormalizedDCG()
        self.auroc = AUROC(task="binary")
        self.ap = AveragePrecision(task="binary")

    def compute_graph_metrics(
        self,
        graph_input: Tensor,
        svos: List[List[Tuple[int, int, int]]],
        indices: Tuple[Tensor, Tensor, Tensor],
    ):
        """
        Params:
            graph_input: (N, k, k, k) tensor of graph probabilities p(x_i -> x_j | x_k)
            svos: lists of SVO triplets for each batch instance
            indices: a tuple of (batch_idxs, phrase_emb_idxs, input_emb_idxs)
        """

        batch_size = graph_input.size(0)
        batch_idxs, phrase_emb_idxs, input_emb_idxs = indices
        graph_target = torch.full_like(graph_input, 0, dtype=torch.long)
        emb_idx_mapping = {}

        # create a mapping dict from SVO phrase indices to embedding indices
        for batch_idx, phrase_emb_idx, input_emb_idx in zip(
            batch_idxs.tolist(), phrase_emb_idxs.tolist(), input_emb_idxs.tolist()
        ):
            if phrase_emb_idx not in emb_idx_mapping:
                emb_idx_mapping[phrase_emb_idx] = {}

            emb_idx_mapping[phrase_emb_idx][batch_idx] = input_emb_idx

        # assign ground truth labels to the target adjacency cube tensor
        for i, svos_i in enumerate(svos):
            for s, v, o in svos_i:
                graph_target[
                    i,
                    emb_idx_mapping[s][i],
                    emb_idx_mapping[o][i],
                    emb_idx_mapping[v][i],
                ] = 1

        # flatten both input and target
        graph_input, graph_target = graph_input.flatten(), graph_target.flatten()

        return dict(
            graph_auroc=self.auroc(graph_input, graph_target),
            graph_ap=self.ap(graph_input, graph_target),
        )

    def compute_retrieval_metrics(
        self,
        input_embeddings: Tensor,
        phrase_embeddings: Tensor,
        positives: List[List[int]],
    ):
        similarities = (
            F.normalize(input_embeddings, dim=-1)
            @ F.normalize(phrase_embeddings, dim=-1).T
        )

        indices, preds, gts = zip(
            *[
                (i, val, j in pos)
                for i, (sim, pos) in enumerate(zip(similarities, positives))
                for j, val in enumerate(sim.T.max(-1).values)
                if val > 0.0
            ]
        )
        indices, preds, gts = (
            torch.tensor(indices, device=input_embeddings.device),
            torch.stack(preds),
            torch.tensor(gts, device=input_embeddings.device),
        )

        return dict(
            retrieval_map=self.rmap(preds, gts, indexes=indices),
            retrieval_mrr=self.rmrr(preds, gts, indexes=indices),
            retrieval_dcg=self.rdcg(preds, gts, indexes=indices),
        )

    def forward(
        self,
        input_embeddings: Tensor,
        phrase_embeddings: Tensor,
        graph_probs: Tensor,
        positives: List[List[int]],
        svos: List[List[Tuple[int, int, int]]],
    ):
        """
        Params:
            input_embeddings: (N, k, d) tensor of input embeddings
            phrase_embeddings: (m, d) tensor of phrase embeddings
            graph_probs: (N, k, k, k) tensor of graph probabilities
            positives: list of lists of positive indices for each batch instance
            svos: list of lists of SVO triplets for each batch instance

        Returns:
            metrics: dict of metrics (graph_auroc)
        """
        indices = self.get_indices(phrase_embeddings, input_embeddings, positives)
        metrics = {}
        metrics.update(self.compute_graph_metrics(graph_probs, svos, indices))
        metrics.update(
            self.compute_retrieval_metrics(
                input_embeddings, phrase_embeddings, positives
            )
        )

        return metrics


# Metrics()(img_embs, pemb, graph_probs, positives, svos)
