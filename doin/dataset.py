import os
import re
import torch
import clip
from dataclasses import dataclass
from typing import List, Tuple, Any
from PIL import Image
from doin.preprocess import preprocess


@dataclass
class SVOItem:
    caption: str
    img_name: str
    phrases: List[str]
    svos: List[Tuple[int, int, int]]


class SVODataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Any], img_dir: str, transform=preprocess):
        super().__init__()

        self.data = data
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def get_svo_item(self, data) -> SVOItem:
        raise NotImplementedError()

    def __getitem__(self, idx):
        item = self.get_svo_item(self.data[idx])
        img = self.transform(Image.open(os.path.join(self.img_dir, item.img_name)))
        return img, item.caption, item.phrases, torch.tensor(item.svos)


_det_regex = re.compile(r"^(the|a|an|his|her|their|my|our|your|its)\s")


def normalize_phrase(phrase: str) -> str:
    return _det_regex.sub("", phrase.lower())


def svo_collate(batch):
    img, captions, positives, svos = zip(*batch)

    # concatenate and lowercase all positivies, and remove duplicates
    pos_set = list(set([normalize_phrase(p) for ps in positives for p in ps]))
    # pos_set = list(set(sum(positives, [])))

    # create a map between a given phrase and its index in pos_set
    pos_map = {s: i for i, s in enumerate(pos_set)}

    # create a list of positives for each instance, as indices of
    # positive phrases in pos_set (which is actually a list)
    pos_xs = []
    svo_xs = []

    for pos, svo in zip(positives, svos):
        pos_xs.append([pos_map[normalize_phrase(p)] for p in pos])
        svo_xs.append(
            [
                (
                    pos_map[normalize_phrase(pos[s.item()])],
                    pos_map[normalize_phrase(pos[v.item()])],
                    pos_map[normalize_phrase(pos[o.item()])],
                )
                for s, v, o in svo
            ]
        )

    return (
        torch.stack(img),
        clip.tokenize(captions),
        clip.tokenize(pos_set),
        pos_xs,
        svo_xs,
    )
