# Copied and modified from https://github.com/lucidrains/voicebox-pytorch/blob/main/voicebox_pytorch/voicebox_pytorch.py

# MIT License
#
# Copyright (c) 2023 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional, Union

import einx
import torch
import torch.nn.functional as F
from einops import pack, rearrange
from torch import nn
from transformers import FastSpeech2ConformerHifiGan, FastSpeech2ConformerHifiGanConfig, PreTrainedModel
from transformers.models.fastspeech2_conformer.modeling_fastspeech2_conformer import length_regulator

from .configs import ConditionalFlowMatchingConfig, ConditionalFlowMatchingWithHifiGanConfig


def exists(val):
    return val is not None


class RandomFourierEmbed(nn.Module):
    """
    Copied from https://github.com/lucidrains/e2-tts-pytorch/blob/main/e2_tts_pytorch/e2_tts.py
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        assert hidden_size % 2 == 0
        self.register_buffer("weights", torch.randn(hidden_size // 2))

    def forward(self, x):
        freqs = einx.multiply("i, j -> i j", x, self.weights) * 2 * torch.pi
        fourier_embed, _ = pack((x, freqs.sin(), freqs.cos()), "b *")
        return fourier_embed


class RotaryEmbedding(nn.Module):
    """
    rotary positional embeddings
    https://arxiv.org/abs/2104.09864
    """

    def __init__(self, hidden_size: int, theta=10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, t: Union[int, torch.Tensor]):
        if not torch.is_tensor(t):
            t = torch.arange(t, device=self.device)

        t = t.type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.amp.autocast("cuda", enabled=False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()


class ConvPositionEmbed(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 31, groups: int = 1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.dw_conv1d = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=groups, padding=kernel_size // 2), nn.GELU()
        )

    def forward(self, x, mask=None):
        if exists(mask):
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = rearrange(x, "b n c -> b c n")
        x = self.dw_conv1d(x)
        out = rearrange(x, "b c n -> b n c")

        if exists(mask):
            out = out.masked_fill(~mask, 0.0)

        return out


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = hidden_size**0.5
        self.to_weight = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.to_weight.weight)

    def forward(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, "b d -> b 1 d")

        normed = F.normalize(x, dim=-1)
        gamma = self.to_weight(condition)
        return normed * self.scale * (gamma + 1.0)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.dropout = dropout

        self.to_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, mask=None, rotary_emb=None):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        if mask is not None and mask.ndim != 4:
            mask = mask.unsqueeze(1).unsqueeze(2)

        _, heads, q_len, _ = q.shape

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L
        if mask is not None:
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention
        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SIGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-2)
        return F.silu(gate) * x


class FeedForward(nn.Module):
    """
    Multi-layered conv1d for Transformer block with a GLU activation function.
    https://arxiv.org/abs/1905.09263
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0, kernel_size: int = 3):
        super().__init__()

        self.conv1 = nn.Conv1d(
            hidden_size, intermediate_size * 2, kernel_size, stride=1, padding=(kernel_size - 1) // 2
        )
        self.glu = SIGLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(intermediate_size, hidden_size, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, hidden_states: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Batch of input tensors.

        Returns:
            `torch.Tensor`: Batch of output tensors `(batch_size, sequence_length, hidden_size)`.
        """
        hidden_states = hidden_states.transpose(-1, 1)

        if mask is not None:
            mask = mask.unsqueeze(1)
            hidden_states = hidden_states.masked_fill(~mask, 0.0)

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.glu(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if mask is not None:
            hidden_states = hidden_states.masked_fill(~mask, 0.0)

        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states.transpose(-1, 1)
        return hidden_states


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        depth: int,
        heads: int,
        intermediate_size: int,
        attn_dropout: float,
        ff_dropout: float,
        use_unet_skip_connection: bool,
    ):
        super().__init__()
        assert depth % 2 == 0
        self.layers = nn.ModuleList([])

        self.rotary_emb = RotaryEmbedding(hidden_size=hidden_size // heads)

        for ind in range(depth):
            layer = ind + 1
            has_skip = use_unet_skip_connection and layer > (depth // 2)

            self.layers.append(
                nn.ModuleList(
                    [
                        nn.Linear(hidden_size * 2, hidden_size, bias=False) if has_skip else None,
                        AdaptiveRMSNorm(hidden_size=hidden_size),
                        Attention(
                            hidden_size=hidden_size,
                            heads=heads,
                            dropout=attn_dropout,
                        ),
                        AdaptiveRMSNorm(hidden_size=hidden_size),
                        FeedForward(hidden_size=hidden_size, intermediate_size=intermediate_size, dropout=ff_dropout),
                    ]
                )
            )

        self.final_norm = nn.RMSNorm(hidden_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, mask=None, adaptive_rmsnorm_cond=None):
        batch, seq_len, *_ = x.shape

        # keep track of skip connections
        skip_connects = []

        # rotary embeddings
        rotary_emb = self.rotary_emb(seq_len)

        # adaptive rmsnorm
        rmsnorm_kwargs = dict()
        if exists(adaptive_rmsnorm_cond):
            rmsnorm_kwargs = dict(condition=adaptive_rmsnorm_cond)

        # going through the attention layers
        for skip_combiner, attn_prenorm, attn, ff_prenorm, ff in self.layers:
            # in the paper, they use a u-net like skip connection
            # unclear how much this helps, as no ablations or further numbers given besides a brief one-two sentence mention

            if not exists(skip_combiner):
                skip_connects.append(x)
            else:
                skip_connect = skip_connects.pop()
                x = torch.cat((x, skip_connect), dim=-1)
                x = skip_combiner(x)

            attn_input = attn_prenorm(x, **rmsnorm_kwargs)
            x = attn(attn_input, mask=mask, rotary_emb=rotary_emb) + x

            ff_input = ff_prenorm(x, **rmsnorm_kwargs)
            x = ff(ff_input, mask=mask) + x

        return self.final_norm(x)


class ConditionalFlowMatchingDurationPredictor(nn.Module):
    """
    Duration predictor module.
    https://arxiv.org/abs/1905.09263
    """

    def __init__(self, config: ConditionalFlowMatchingConfig):
        super().__init__()
        self.log_domain_offset = 1.0
        self.conv = nn.Conv1d(config.dim_cond_emb, 1, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.FloatTensor):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, max_text_length, input_dim)`):
                Batch of input sequences.

        Returns:
            `torch.Tensor`: Batch of predicted durations in log domain `(batch_size, max_text_length)`.

        """
        # (batch_size, input_dim, max_text_length)
        hidden_states = hidden_states.transpose(1, -1)

        # NOTE: calculate in log domain, (batch_size, max_text_length)
        hidden_states = self.conv(hidden_states).squeeze(1)

        if not self.training:
            # NOTE: calculate in linear domain
            hidden_states = torch.clamp(torch.round(hidden_states.exp() - self.log_domain_offset), min=0).long()

        return hidden_states


class ConditionalFlowMatchingModel(PreTrainedModel):
    config_class = ConditionalFlowMatchingConfig

    def __init__(self, config: ConditionalFlowMatchingConfig, embedding: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.config = config

        self.time_cond_mlp = nn.Sequential(
            RandomFourierEmbed(config.hidden_size), nn.Linear(config.hidden_size + 1, config.hidden_size), nn.SiLU()
        )
        self.to_cond_emb = (
            nn.Embedding(config.vocab_size + 1, config.dim_cond_emb, padding_idx=0) if embedding is None else embedding
        )
        self.to_embed = nn.Linear(config.dim_in + config.dim_cond_emb, config.hidden_size)
        self.conv_embed = ConvPositionEmbed(
            hidden_size=config.hidden_size,
            kernel_size=config.conv_pos_embed_kernel_size,
            groups=config.conv_pos_embed_groups,
        )

        self.transformer = Transformer(
            hidden_size=config.hidden_size,
            depth=config.depth,
            heads=config.heads,
            intermediate_size=config.intermediate_size,
            ff_dropout=config.ff_dropout,
            use_unet_skip_connection=config.use_unet_skip_connection,
            attn_dropout=config.attn_dropout,
        )

        self.to_pred = nn.Linear(config.hidden_size, config.dim_in, bias=False)
        self.duration_predictor = ConditionalFlowMatchingDurationPredictor(config) if config.predict_duration else None

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.LongTensor,
        spectrogram_labels: torch.FloatTensor,
        duration_labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            spectrogram_labels (`torch.FloatTensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`):
                Batch of padded target features.
            duration_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`):
                Batch of padded durations.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`):
                Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
                `[0, 1]`: 0 for tokens that are **masked**, 1 for tokens that are **not masked**.
        """
        mask = (spectrogram_labels != -100).any(dim=-1)
        batch, seq_len, _ = spectrogram_labels.shape
        spectrogram_labels = (spectrogram_labels - self.config.mean) / self.config.std

        # main conditional flow logic is below
        x0 = torch.randn_like(spectrogram_labels)
        times = torch.rand((batch,), device=self.device)
        t = times.unsqueeze(1).unsqueeze(2)
        xt = (1 - t) * x0 + t * spectrogram_labels
        ut = spectrogram_labels - x0

        # phoneme or semantic conditioning embedding
        hidden_states = self.to_cond_emb(input_ids)

        # forward duration predictor
        duration_loss = 0
        if duration_labels is not None:
            duration_predictions = self.duration_predictor(hidden_states)
            # use groundtruth in training
            hidden_states = length_regulator(hidden_states, duration_labels)

            attention_mask = attention_mask.bool()
            duration_predictions = duration_predictions.masked_select(attention_mask)
            duration_labels = duration_labels.masked_select(attention_mask)
            duration_labels = torch.log(duration_labels.float() + 1)
            duration_loss = F.mse_loss(duration_predictions, duration_labels)

        hidden_states = torch.cat([xt, hidden_states], dim=-1)

        x = self.to_embed(hidden_states)
        x = self.conv_embed(x, mask=mask) + x

        time_emb = self.time_cond_mlp(times)

        # attend
        x = self.transformer(x, mask=mask, adaptive_rmsnorm_cond=time_emb)
        x = self.to_pred(x)

        return F.mse_loss(x[mask], ut[mask]) + duration_loss

    @torch.inference_mode()
    def sample(
        self,
        input_ids: torch.LongTensor,
        dt: float = 0.1,
        truncation_value: Optional[float] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            dt (`float`, defaults to 0.1):
                Step size for the ordinary differential equation (ODE).
            truncation_value (`float`, *optional*, defaults to `None`):
                Truncation value of a prior sample x0~N(0, 1).
                https://arxiv.org/abs/1809.11096
        Returns:
            x1 (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Synthesized log mel-spectrograms.
        """
        hidden_states = self.to_cond_emb(input_ids)

        # forward duration predictor
        if self.duration_predictor is not None:
            duration_predictions = self.duration_predictor(hidden_states)
            hidden_states = length_regulator(hidden_states, duration_predictions)

        xt = torch.randn(1, hidden_states.shape[1], self.config.dim_in, device=hidden_states.device)
        if truncation_value is not None:
            xt = torch.clamp(xt, -truncation_value, truncation_value)

        for t in torch.arange(0, 1, dt, device=self.device):
            # concat source signal, semantic / phoneme conditioning embed, and conditioning
            # and project
            x = torch.cat([xt, hidden_states], dim=-1)
            x = self.to_embed(x)
            x = self.conv_embed(x) + x

            time_emb = self.time_cond_mlp(t.unsqueeze(0))

            # attend
            x = self.transformer(x, adaptive_rmsnorm_cond=time_emb)
            vt = self.to_pred(x)
            xt = xt + vt * dt

        return xt * self.config.std + self.config.mean


class ConditionalFlowMatchingWithHifiGan(PreTrainedModel):
    config_class = ConditionalFlowMatchingWithHifiGanConfig

    def __init__(self, config: ConditionalFlowMatchingWithHifiGanConfig):
        super().__init__(config)
        self.model = ConditionalFlowMatchingModel(config.model_config)
        self.vocoder = FastSpeech2ConformerHifiGan(config.vocoder_config)

    @classmethod
    def load_pretrained(cls, model_path, vocoder_path) -> "ConditionalFlowMatchingWithHifiGan":
        model_config = ConditionalFlowMatchingConfig.from_pretrained(model_path)
        vocoder_config = FastSpeech2ConformerHifiGanConfig.from_pretrained(vocoder_path)
        config = ConditionalFlowMatchingWithHifiGanConfig(model_config.to_dict(), vocoder_config.to_dict())

        model = cls(config)
        model.model = ConditionalFlowMatchingModel.from_pretrained(model_path)
        model.vocoder = FastSpeech2ConformerHifiGan.from_pretrained(vocoder_path)
        return model

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor,
        dt: float = 0.1,
        truncation_value: Optional[float] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            dt (`float`, defaults to 0.1):
                Step size for the ordinary differential equation (ODE).
            truncation_value (`float`, *optional*, defaults to `None`):
                Truncation value of a prior sample x0~N(0, 1).
                https://arxiv.org/abs/1809.11096
        Returns:
            x1 (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Synthesized log mel-spectrograms.
        """
        spectrogram = self.model.sample(input_ids, dt, truncation_value)
        waveform = self.vocoder(spectrogram)
        return waveform
