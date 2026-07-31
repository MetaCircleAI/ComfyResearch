"""Types that expose ``observable_numpy_arrays()`` with an ``embedding`` key for trainer observables."""

from __future__ import annotations

from comfy_research.engine.models.attention_only_model import AttentionTokenPredictBundle
from comfy_research.engine.models.diagonal_ssm_token_model import DiagonalSsmTokenPredictBundle
from comfy_research.engine.models.hyena_like_conv_model import HyenaLikeConvTokenPredictBundle
from comfy_research.engine.models.linear_attention_model import LinearAttentionTokenPredictBundle
from comfy_research.engine.models.multi_token_transformer_model import MultiTokenTransformerModel
from comfy_research.engine.models.rwkv_time_mix_token_model import RwkvTimeMixTokenPredictBundle
from comfy_research.engine.models.slot_attention_token_model import SlotAttentionTokenPredictBundle
from comfy_research.engine.models.token_transformer_model import TokenTransformerModel

TOKEN_LM_EMBEDDING_OBSERVABLE_MODULES = (
    AttentionTokenPredictBundle,
    LinearAttentionTokenPredictBundle,
    DiagonalSsmTokenPredictBundle,
    RwkvTimeMixTokenPredictBundle,
    HyenaLikeConvTokenPredictBundle,
    SlotAttentionTokenPredictBundle,
    TokenTransformerModel,
    MultiTokenTransformerModel,
)
