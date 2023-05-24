import clip
from typing import Literal
from doin.model import DOIN


def build_model_from_clip(
    clip_model_type: Literal["ViT-B/32"] = "ViT-B/32",
    k_embeddings: int = 20,
):
    """
    Builds a DOIN model with weights from the CLIP model
    """
    clip_model, _ = clip.load(clip_model_type, device="cpu", jit=False)
    state_dict = clip_model.state_dict()
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [
            k
            for k in state_dict.keys()
            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ]
    )
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")
        )
    )

    model = DOIN(
        k_embeddings,
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
    )
    doin_state_dict = model.state_dict()

    for k, p in state_dict.items():
        # these are modules/params that are moved into the LanguageTransformer module from CLIP
        lang_params = ["transformer", "token_embedding", "ln_final", "text_projection"]

        for lang_param in lang_params:
            if k[: len(lang_param)] == lang_param:
                doin_state_dict[f"language.{k}"] = p

        doin_state_dict["language.positional_embedding"][:context_length] = state_dict[
            "positional_embedding"
        ]
        doin_state_dict["language.text_projection"] = state_dict["text_projection"]

        visual_skip = ["visual.positional_embedding", "visual.class_embedding"]

        if "visual" in k and k not in visual_skip:
            doin_state_dict[k] = p

        # use the clip class token for vision, but repeat it for all k class tokens
        doin_state_dict["visual.class_embedding"] = (
            state_dict["visual.class_embedding"].unsqueeze(0).repeat(k_embeddings, 1)
        )
        doin_state_dict["logit_scale"] = state_dict["logit_scale"]

    model.load_state_dict(doin_state_dict)

    return model.eval()
