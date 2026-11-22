"""Conditional-VAE ACT with one continuous SO-101 joint-target head."""
from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d

from .transformer import DecoderLayer, EncoderLayer, sine_1d, sine_2d

KL_WEIGHT = 10.0


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int = 6
    cameras: tuple = ("wrist", "front")
    image_hw: tuple = (240, 320)
    chunk_size: int = 32
    hidden_dim: int = 256
    nheads: int = 8
    dim_feedforward: int = 1024
    encoder_layers: int = 4
    decoder_layers: int = 4
    posterior_layers: int = 3
    latent_dim: int = 32
    dropout: float = 0.1


class ACT(nn.Module):
    def __init__(self, cfg, *, pretrained):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.visual_projection = nn.Conv2d(512, d, 1)
        self.state_projection = nn.Linear(cfg.state_dim, d)
        self.latent_projection = nn.Linear(cfg.latent_dim, d)
        self.camera_pos = nn.Embedding(len(cfg.cameras), d)
        self.type_pos = nn.Embedding(3, d)
        self.queries = nn.Embedding(cfg.chunk_size, d)
        self.encoder = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.encoder_layers))
        self.decoder = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.decoder_layers))
        self.action_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.state_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.posterior_state = nn.Linear(cfg.state_dim, d)
        self.posterior_action = nn.Linear(cfg.state_dim, d)
        self.posterior = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.posterior_layers))
        self.moments = nn.Linear(d, 2 * cfg.latent_dim)
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1))

    def forward(self, images, state, actions=None, pad=None, *, generator=None):
        b = state.shape[0]
        cfg = self.cfg
        if (images.shape != (b, len(cfg.cameras), 3, *cfg.image_hw)
                or state.shape != (b, cfg.state_dim)):
            raise ValueError("ACT input must match the checkpoint's cameras, resolution and joint order")
        if (actions is None) != (pad is None):
            raise ValueError("Provide action targets and their padding mask together")
        if actions is not None and (actions.shape != (b, cfg.chunk_size, cfg.state_dim)
                                    or pad.shape != (b, cfg.chunk_size) or pad.dtype != torch.bool):
            raise ValueError("Invalid action chunk or padding mask")
        if actions is None:
            z = state.new_zeros(b, cfg.latent_dim)
            mean = logvar = None
        else:
            tokens = torch.cat((self.cls.expand(b, -1, -1), self.posterior_state(state)[:, None],
                                self.posterior_action(actions)), dim=1)
            pos = sine_1d(torch.arange(tokens.size(1), device=state.device), cfg.hidden_dim)[None]
            mask = torch.cat((pad.new_zeros(b, 2), pad), dim=1)
            for layer in self.posterior:
                tokens = layer(tokens, pos, mask)
            mean, logvar = self.moments(tokens[:, 0]).chunk(2, dim=-1)
            noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
            z = mean + noise * (0.5 * logvar).exp()

        pixels = (images.float() / 255.0 - self.rgb_mean) / self.rgb_std
        visual = self.visual_projection(self.backbone(pixels.flatten(0, 1)))
        _, d, h, w = visual.shape
        visual = visual.flatten(2).transpose(1, 2).reshape(b, -1, d)
        spatial = sine_2d(h, w, d, state.device)
        image_pos = (spatial[None] + self.camera_pos.weight[:, None] + self.type_pos.weight[2]).reshape(1, -1, d)
        pos = torch.cat((self.type_pos.weight[None, :2], image_pos), dim=1)
        memory = torch.cat((self.state_projection(state)[:, None], self.latent_projection(z)[:, None], visual), dim=1)
        for layer in self.encoder:
            memory = layer(memory, pos)
        queries = self.queries.weight[None].expand(b, -1, -1)
        output = torch.zeros_like(queries)
        for layer in self.decoder:
            output = layer(output, queries, memory, pos)
        return self.action_head(output), mean, logvar


def objective(prediction, actions, pad, mean, logvar):
    valid = (~pad).unsqueeze(-1)
    l1 = ((prediction - actions).abs() * valid).sum() / (valid.sum() * actions.size(-1))
    kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(dim=1).mean()
    return l1 + KL_WEIGHT * kl, l1, kl
