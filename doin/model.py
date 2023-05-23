import torch
import numpy as np
from typing import Tuple, Union
from torch import Tensor, nn
from clip.model import Transformer, LayerNorm


class MLP(nn.Module):
    def __init__(self, d=512, out_dim=512, num_layers=3):
        super().__init__()

        layers = []

        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(nn.Linear(d, out_dim))
            else:
                layers.append(nn.Linear(d, d))
                layers.append(nn.ReLU(inplace=True))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class VisionTransformer(nn.Module):
    """
    Modified vision transformer, original code at <https://github.com/openai/CLIP>
    Prepends k "class tokens" which will constitute the concept embeddings
    """

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        k_embeddings: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.k_embeddings = k_embeddings
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(k_embeddings, width))
        self.positional_embedding = nn.Parameter(
            scale
            * torch.randn((input_resolution // patch_size) ** 2 + k_embeddings, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0],
                    self.k_embeddings,
                    x.shape[-1],
                    dtype=x.dtype,
                    device=x.device,
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, : self.k_embeddings, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class LanguageTransformer(nn.Module):
    """
    Extends the plain transformer used in CLIP to append similar tokens to the vision transformer
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        k_embeddings: int,
        context_length: int,
        transformer_width: int,
        transformer_layers: int,
        transformer_heads: int,
    ):
        super().__init__()

        self.k_embeddings = k_embeddings
        self.context_length = context_length
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attn_mask(),
        )
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.randn(context_length + k_embeddings, transformer_width)
        )
        self.class_embedding = nn.Parameter(
            torch.randn(k_embeddings, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def build_attn_mask(self):
        # adopted from CLIP
        # lazily create causal attention mask, ignore
        # pytorch uses additive attention mask; fill with -inf
        n = self.context_length + self.k_embeddings
        mask = torch.empty(n, n)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text: Tensor):
        x = self.token_embedding(text)

        # append class tokens (not prepend)
        x = torch.cat(
            (
                x,
                self.class_embedding
                + torch.zeros(
                    x.shape[0],
                    self.k_embeddings,
                    x.shape[-1],
                    dtype=x.dtype,
                    device=x.device,
                ),
            ),
            1,
        )

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        # the text embedding from the eot token (i.e., the token with the highest token index)
        single = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        k_embs = x[:, self.context_length :] @ self.text_projection

        return single, k_embs


class GraphDecoder(nn.Module):
    def __init__(self, d=512, num_layers=3):
        super().__init__()

        # Encoder that takes in k d-dimensional embeddings and
        # applies multi head attention to highlight the embeddings
        # that are the source or target of an edge
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d, nhead=8, dim_feedforward=4 * d),
            num_layers,
            nn.LayerNorm(d),
        )
        self.src_proj = MLP(d, d, 3)
        self.tgt_proj = MLP(d, d, 3)
        self.cond_proj = MLP(d, d, 3)
        self.reduce_proj = MLP(d, 1, 3)

    def forward(self, x):
        x = self.encoder(x)
        src = self.src_proj(x)
        tgt = self.tgt_proj(x)
        cond = self.cond_proj(x)

        outer_prod = src @ tgt.transpose(-2, -1)

        p_x = outer_prod.unsqueeze(-1).unsqueeze(-1)
        cond = cond.unsqueeze(1).unsqueeze(1)
        p_x = p_x * cond
        p_x = self.reduce_proj(p_x).squeeze(-1)

        return p_x


class GraphDecoder(nn.Module):
    def __init__(self, d=512, num_layers=3):
        super().__init__()

        # Encoder that takes in k d-dimensional embeddings and
        # applies multi head attention to highlight the embeddings
        # that are the source or target of an edge
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d, nhead=8, dim_feedforward=4 * d),
            num_layers,
            nn.LayerNorm(d),
        )
        self.sub_proj = MLP(d, d, 3)
        self.obj_proj = MLP(d, d, 3)
        self.verb_proj = MLP(d, d, 3)

    def forward(self, x):
        x = self.encoder(x)

        sub = self.sub_proj(x)
        obj = self.obj_proj(x)
        verb = self.verb_proj(x)

        sub = sub.unsqueeze(1) * verb.unsqueeze(2)
        obj = obj.unsqueeze(1) * verb.unsqueeze(2)

        x = sub @ obj.transpose(-2, -1)

        return x


class DOIN(nn.Module):
    def __init__(
        self,
        k_embeddings: int,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        # graph
        graph_decoder_layers: int,
    ):
        super().__init__()

        # vision
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            k_embeddings=k_embeddings,
        )

        # text
        self.language = LanguageTransformer(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            k_embeddings=k_embeddings,
            context_length=context_length,
            transformer_width=transformer_width,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # interaction graph
        self.graph_decoder = GraphDecoder(embed_dim, graph_decoder_layers)

    def encode_image(self, image: Tensor):
        return self.visual(image)

    def encode_text(self, text: Tensor):
        return self.language(text)[1]
        return single[1]

    def encode_phrase(self, text: Tensor):
        return self.language(text)[0]

    def forward(self, images, texts, phrases):
        img_feats = self.encode_image(images)
        txt_feats = self.encode_text(texts)
        phrase_feats = self.encode_phrase(phrases)
        img_graph_feats = self.graph_decoder(img_feats)
        txt_graph_feats = self.graph_decoder(txt_feats)

        return dict(
            img=img_feats,
            txt=txt_feats,
            phrases=phrase_feats,
            img_graph=img_graph_feats,
            txt_graph=txt_graph_feats,
        )
