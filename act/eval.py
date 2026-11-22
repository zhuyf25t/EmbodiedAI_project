"""Independent simulation processes share one batched inference model."""
from contextlib import AbstractContextManager
import multiprocessing as mp

import numpy as np

from .sim_worker import run_worker

VALIDATION_SEEDS = tuple(range(500, 520))
EXECUTION_HORIZON = 8


class SimPool(AbstractContextManager):
    def __init__(self, workers, image_hw):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers, self.image_hw = workers, image_hw
        self.slots = []

    def __enter__(self):
        ctx = mp.get_context("spawn")
        try:
            for _ in range(self.workers):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=run_worker, args=(child, self.image_hw))
                proc.start()
                child.close()
                self.slots.append((parent, proc))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for pipe, _ in self.slots:
            try:
                pipe.send(("close", None))
            except (EOFError, OSError):
                pass
            pipe.close()
        for _, proc in self.slots:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join()
        self.slots.clear()

    def evaluate(self, model, stats, seeds, device):
        import torch

        seeds = list(seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Evaluation needs distinct seeds")
        model.eval()
        norms = {k: torch.as_tensor(v, device=device) for k, v in stats.items()}
        results = []
        upcoming = iter(seeds)
        active = {}

        def assign(slot):
            seed = next(upcoming, None)
            if seed is not None:
                self.slots[slot][0].send(("reset", seed))
                active[slot] = seed

        for slot in range(min(len(self.slots), len(seeds))):
            assign(slot)
        with torch.inference_mode():
            while active:
                inputs = {}
                for slot, seed in list(active.items()):
                    status, response = self.slots[slot][0].recv()
                    if status != "ok":
                        raise RuntimeError(f"Simulation failed for seed {seed}: {response}")
                    obs, result = response
                    if result is None:
                        inputs[slot] = obs
                    else:
                        if result["seed"] != seed:
                            raise RuntimeError("Worker returned a different seed")
                        results.append(result)
                        del active[slot]
                        assign(slot)
                if inputs:
                    images, states = zip(*inputs.values())
                    images = torch.as_tensor(np.stack(images), device=device)
                    states = torch.as_tensor(np.stack(states), device=device)
                    states = (states - norms["state_mean"]) / norms["state_std"]
                    chunk, _, _ = model(images, states)
                    targets = chunk[:, :EXECUTION_HORIZON] * norms["action_std"] + norms["action_mean"]
                    for slot, target in zip(inputs, targets.cpu().numpy()):
                        self.slots[slot][0].send(("advance", target))
        by_seed = {e["seed"]: e for e in results}
        if len(results) != len(seeds) or set(by_seed) != set(seeds):
            raise RuntimeError("Incomplete evaluation")
        return [by_seed[s] for s in seeds]


def summary(episodes):
    successes = sum(e["success"] for e in episodes)
    return dict(successes=successes, total=len(episodes), success_rate=successes / len(episodes),
                episodes=[{k: v for k, v in e.items() if k != "trace"} for e in episodes])
