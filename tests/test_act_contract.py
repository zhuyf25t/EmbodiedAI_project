"""Regression checks for padding and closed-loop scoring; no training required."""
import unittest

import torch

from act.eval import summary
from act.model import ACT, ModelConfig, objective


class ACTContractTest(unittest.TestCase):
    def test_padding_is_invisible_to_posterior_and_loss(self):
        cfg = ModelConfig(image_hw=(32, 32), hidden_dim=32, nheads=4, dim_feedforward=64,
                          chunk_size=4, encoder_layers=1, decoder_layers=1, posterior_layers=1)
        torch.manual_seed(5)
        model = ACT(cfg, pretrained=False).eval()
        images = torch.zeros(2, 2, 3, 32, 32, dtype=torch.uint8)
        states = torch.zeros(2, cfg.state_dim)
        actions = torch.randn(2, 4, cfg.state_dim)
        pad = torch.tensor([[False, False, True, True], [False, True, True, True]])
        with torch.inference_mode():
            reference = None
            for labels in (actions, actions.masked_fill(pad[..., None], 1000)):
                torch.manual_seed(7)
                pred, mean, logvar = model(images, states, labels, pad)
                values = (mean, logvar, *objective(pred, labels, pad, mean, logvar))
                if reference is not None:
                    for before, after in zip(reference, values):
                        torch.testing.assert_close(before, after)
                reference = values
            torch.testing.assert_close(model(images, states)[0], model(images, states)[0])

    def test_bilateral_contact_alone_does_not_raise_selection_score(self):
        episodes = [dict(success=False, ever_bilateral=True), dict(success=True, ever_bilateral=True)]
        report = summary(episodes)
        self.assertEqual((report["successes"], report["total"], report["success_rate"]), (1, 2, 0.5))


if __name__ == "__main__":
    unittest.main()
