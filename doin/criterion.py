import torch
import torch.nn.functional as F
from typing import List, Tuple, Union
from torch import Tensor, nn
from torchvision.ops import sigmoid_focal_loss
from scipy.optimize import linear_sum_assignment
from doin.matcher import Matcher


class GraphLoss(nn.Module):
    def __init__(self, gamma=4, alpha=0.75):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(
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
        graph_target = torch.full_like(graph_input, 0)
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

        # compute the loss between the input graph and the target graph
        return dict(
            graph=sigmoid_focal_loss(
                graph_input,
                graph_target,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction="sum",
            )
            / graph_target.sum()
        )


class AlignmentLoss(nn.Module):
    def __init__(self, margin=1, num_hard_negatives=3, num_random_negatives=2):
        super().__init__()

        self.num_hard_negatives = num_hard_negatives
        self.num_random_negatives = num_random_negatives

        self.ce = nn.CrossEntropyLoss()
        self.triplet = nn.TripletMarginWithDistanceLoss(
            distance_function=self.get_distance,
            margin=margin,
        )

    def get_distance(self, a, b):
        return 1 - F.cosine_similarity(a, b, dim=-1)

    def get_triplet_loss(
        self,
        phrase_embeddings: Tensor,
        input_embeddings: Tensor,
        similarity_matrix: Tensor,
        indices: Tuple[Tensor, Tensor, Tensor],
    ):
        """
        Params:
            phrase_embeddings: (m, d) tensor of phrase embeddings
            input_embeddings: (N, k, d) tensor of input embeddings
            similarity_matrix: (N, m, k) tensor of cosine similarities between phrase and input embeddings
            indices: a tuple of (batch_idxs, phrase_emb_idxs, input_emb_idxs)
        """
        batch_idxs, phrase_emb_idxs, input_emb_idxs = indices
        num_negatives = self.num_hard_negatives + self.num_random_negatives
        k_embeddings = similarity_matrix.size(-1)

        # clone the similarity matrix and set the positive examples to -inf such that they are not selected as negatives
        neg_sim = similarity_matrix.clone()
        neg_sim[batch_idxs, phrase_emb_idxs, input_emb_idxs] = float("-inf")
        neg_embs = [[] for _ in range(num_negatives)]

        # select the top num_hard_negatives hard negatives for each positive example
        for batch_idx, input_idxs in zip(
            batch_idxs,
            neg_sim[batch_idxs, phrase_emb_idxs]
            .topk(self.num_hard_negatives, -1)
            .indices,
        ):
            for k, input_idx in enumerate(input_idxs):
                neg_embs[k].append(input_embeddings[batch_idx, input_idx])

        # select num_random_negatives random negatives for each positive example
        for batch_idx, phrase_emb_idx, input_emb_idx in zip(
            batch_idxs, phrase_emb_idxs, input_emb_idxs
        ):
            rand_idxs = torch.randperm(k_embeddings - 1, device=input_embeddings.device)
            rand_idxs = rand_idxs[rand_idxs != input_emb_idx][
                : self.num_random_negatives
            ]

            for k, rand_idx in zip(
                range(self.num_hard_negatives, num_negatives), rand_idxs
            ):
                neg_embs[k].append(input_embeddings[batch_idx, rand_idx])

        neg_embs = torch.stack(sum(neg_embs, []))

        return self.triplet(
            phrase_embeddings[phrase_emb_idxs].repeat(num_negatives, 1),
            input_embeddings[batch_idxs, input_emb_idxs].repeat(num_negatives, 1),
            neg_embs,
        )

    def get_cross_entropy_loss(
        self,
        similarity_matrix: Tensor,
        indices: Tuple[Tensor, Tensor, Tensor],
        temperature: Union[float, Tensor],
    ):
        """
        Params:
            similarity_matrix: (N, m, k) tensor of cosine similarities between phrase and input embeddings
            indices: a tuple of (batch_idxs, phrase_emb_idxs, input_emb_idxs)
            temperature: temperature for the softmax function (in CE)
        """
        batch_idxs, phrase_emb_idxs, input_emb_idxs = indices
        loss = self.ce(
            similarity_matrix[batch_idxs, :, input_emb_idxs] * temperature,
            phrase_emb_idxs,
        )

        return loss

    def forward(
        self,
        phrase_embeddings: Tensor,
        input_embeddings: Tensor,
        indices: Tuple[Tensor, Tensor, Tensor],
        temperature: Union[float, Tensor],
    ):
        """
        Params:
            phrase_embeddings: (m, d) tensor of phrase embeddings
            input_embeddings: (N, k, d) tensor of input embeddings
            indices: a tuple of (batch_idxs, phrase_emb_idxs, input_emb_idxs)
            temperature: temperature for the softmax function (in CE)
        """

        # compute the similarity matrix between the phrase and input embeddings
        similarity_matrix = (
            F.normalize(input_embeddings, dim=-1)
            @ F.normalize(phrase_embeddings, dim=-1).t()
        ).transpose(-2, -1)

        return dict(
            triplet=self.get_triplet_loss(
                phrase_embeddings, input_embeddings, similarity_matrix, indices
            ),
            ce=self.get_cross_entropy_loss(similarity_matrix, indices, temperature),
        )


class SetAlignmentGraphLoss(nn.Module, Matcher):
    def __init__(self):
        super().__init__()

        self.align = AlignmentLoss()
        self.graph = GraphLoss()

    def forward(
        self,
        input_embeddings: Tensor,
        phrase_embeddings: Tensor,
        graph_probs: Tensor,
        positives: List[List[int]],
        svos: List[List[Tuple[int, int, int]]],
        temperature: Union[Tensor, float],
    ):
        """
        Params:
            input_embeddings: (N, k, d) tensor of input embeddings
            phrase_embeddings: (m, d) tensor of phrase embeddings
            graph_probs: (N, k, k, k) tensor of graph probabilities
            positives: list of lists of positive indices for each batch instance
            svos: list of lists of SVO triplets for each batch instance
            temperature: temperature for the softmax function (in CE)

        Returns:
            losses: dict of losses (triplet, ce, graph)
        """
        indices = self.get_indices(phrase_embeddings, input_embeddings, positives)
        losses = {}

        losses.update(
            self.align(phrase_embeddings, input_embeddings, indices, temperature)
        )
        losses.update(self.graph(graph_probs, svos, indices))

        return losses
