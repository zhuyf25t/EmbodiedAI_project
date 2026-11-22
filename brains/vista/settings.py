"""Common VISTA tool bounds. Only the provider is configured by the operator."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


def local_values(path=None):
    return dotenv_values(path or Path(__file__).resolve().parents[2] / ".env",
                         encoding="utf-8-sig", interpolate=False)


def setting(names, values, environ, default=None):
    for source in (environ, values):
        for name in names:
            if name in source:
                return source[name]
    return default


def positive_integer(value, name):
    if not str(value).strip().isdigit() or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class VistaSettings:
    max_model_calls: int = 12
    max_tool_calls: int = 8
    max_actions: int = 16
    max_views: int = 4
    inspect_edge: int = 1024
    max_samples: int = 1024
    max_question_chars: int = 1024
    max_label_chars: int = 128
    max_guide_chars: int = 8000
    max_working_chars: int = 4000
    history_default_limit: int = 10
    history_max_limit: int = 50
    history_max_observations: int = 500

    @classmethod
    def from_env(cls):
        value = setting(("max_model_calls",), local_values(), os.environ, 12)
        return cls(max_model_calls=positive_integer(value, "max_model_calls"))
