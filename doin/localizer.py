import torch
import torchvision
import torch.nn.functional as F
from torch import Tensor, nn
from scipy.optimize import linear_sum_assignment


class DOINLocalizationPostProcessor(nn.Module):
    """
    A post-processor for DOIN that localizes objects in the image using an OTS object detector model

    Args:
        det_model: the object detection model
        patch_size: the relative patch size of the DOIN model (i.e., patch size / image size)
        img_dim: the image dimension for localization (expects square images, not necessarily the same as the DOIN image dimension)
        threshold: the confidence threshold for object detection
    """

    def __init__(
        self,
        det_model,
        k_embeddings: int,
        patch_size: float,
        img_dim: int,
        threshold: float = 0.5,
    ):
        super().__init__()

        self.det_model = det_model
        self.k_embeddings = k_embeddings
        self.patch_size = patch_size
        self.img_dim = img_dim
        self.threshold = threshold

        patch_side_n = int(
            1 / self.patch_size
        )  # number of patches per side, assuming square images

        self.register_buffer(
            "patch_boxes",
            torch.tensor(
                [
                    [
                        j * self.patch_size,
                        i * self.patch_size,
                        (j + 1) * self.patch_size,
                        (i + 1) * self.patch_size,
                    ]
                    for i in range(patch_side_n)
                    for j in range(patch_side_n)
                ]
            ),
        )

    def get_obj_loc(self, img: Tensor):
        """
        Returns the bounding boxes and scores of objects in an image, assuming the model
        follows the typical torchvision object detection API.

        Args:
            model: the object detection model
            img: the image tensor
            img_dim: the image dimension (expects square images)
            threshold: the confidence threshold for object detection
        """

        preds = self.det_model([img])[0]

        bboxes, scores = zip(
            *[
                (box.detach() / self.img_dim, score.detach())
                for box, score in zip(preds["boxes"], preds["scores"])
                if score > self.threshold
            ]
        )

        return torch.stack(bboxes), torch.stack(scores)

    def get_patch_masks(self, bboxes: Tensor, min_area_overlap: float = 0.5):
        masks = []

        for bbox in bboxes:
            inner_patch_areas = torchvision.ops.box_area(
                torch.cat(
                    (
                        torch.maximum(bbox[:2], self.patch_boxes[:, :2]),
                        torch.minimum(bbox[2:], self.patch_boxes[:, 2:]),
                    ),
                    1,
                )
            ).clamp(0) / (self.patch_size**2)

            masks.append(inner_patch_areas > min_area_overlap)

        return torch.stack(masks)

    def __call__(
        self,
        img: Tensor,
        img_embeddings: Tensor,
        img_attention_weights: Tensor,
        phrase_embeddings: Tensor,
    ):
        """
        Returns the bounding boxes and scores of objects in an image, assuming the model
        follows the typical torchvision object detection API.

        Args:
            img: the image tensor
            img_embeddings: the DOIN image embeddings
            phrase_embeddings: the DOIN phrase embeddings
        """

        bboxes, scores = self.get_obj_loc(img)
        masks = self.get_patch_masks(bboxes)
        attn_weights = img_attention_weights.permute(1, 0, 2, 3)
        similarities = (
            F.normalize(img_embeddings, dim=-1)
            @ F.normalize(phrase_embeddings, dim=-1).T
        )

        out = []
        foci = []

        if bboxes.shape[0] == 0:
            return out

        for bbox, mask in zip(bboxes, masks):
            inside_attn = attn_weights.cumsum(dim=1)[
                0, -1, : self.k_embeddings, self.k_embeddings :
            ][:, mask].sum(-1)
            # outside_attn = attn_weights.cumsum(dim=1)[
            #     0, -1, : self.k_embeddings, self.k_embeddings :
            # ][:, ~mask].sum(-1)
            embedding_focus = inside_attn  # / outside_attn
            foci.append(embedding_focus)

        foci = torch.stack(foci)

        if torch.isnan(foci).any() or torch.isinf(foci).any():
            return out

        _, img_emb_idxs = linear_sum_assignment(-foci.cpu().numpy())

        for img_embedding_idx, bbox, score in zip(img_emb_idxs, bboxes, scores):
            # img_embedding_idx = embedding_focus.argmax(-1)
            phrase_similarity, phrase_idx = similarities[img_embedding_idx].max(-1)

            out.append(
                dict(
                    bbox=bbox,
                    score=score,
                    img_embedding_idx=img_embedding_idx,
                    phrase_embedding_idx=phrase_idx,
                    phrase_similarity=phrase_similarity.detach(),
                )
            )

        return out
