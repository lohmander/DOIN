import torch
from typing import List
from torch import Tensor
from scipy.optimize import linear_sum_assignment


class Matcher:
    @torch.no_grad()
    def get_indices(
        self,
        phrase_embeddings: Tensor,
        input_embeddings: Tensor,
        positives: List[List[int]],
    ):
        """
        Get the optimal alignment between the phrase and input embeddings using the Hungarian algorithm.

        Params:
            phrase_embeddings: (m, d) tensor of phrase embeddings
            input_embeddings: (N, k, d) tensor of input embeddings
            positives: list of lists of positive indices for each batch instance
        """

        # compute the cosine similarity between the phrase and input embeddings, and flip the sign to get the cost matrix
        C = -(input_embeddings @ phrase_embeddings.t()).transpose(-2, -1)

        # compute the optimal alignment between the phrase and input embeddings using the Hungarian algorithm
        indices = [
            linear_sum_assignment(cs_i[pos_i].cpu().detach())
            for i, (cs_i, pos_i) in enumerate(zip(C, positives))
        ]

        # convert the indices to tensors
        batch_idxs, phrase_emb_idxs, input_emb_idxs = torch.cat(
            [
                torch.stack(
                    [
                        torch.full((len(pos_i),), i, device=C.device),
                        torch.as_tensor(pos_i, device=C.device),
                        torch.as_tensor(idx[1], device=C.device),
                    ]
                )
                for i, (idx, pos_i) in enumerate(zip(indices, positives))
            ],
            1,
        ).long()

        return batch_idxs, phrase_emb_idxs, input_emb_idxs
