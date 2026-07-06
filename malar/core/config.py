"""Central configuration. All env-driven; no provider/model strings live in code paths."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


@dataclass
class Settings:
    # Neo4j
    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "malar_dev_password"))

    # Qdrant
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))

    # LLM gateway (aliases only — provider chosen in infra/litellm_config.yaml)
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:4000/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "sk-malar-local"))
    alias_reasoner: str = field(default_factory=lambda: _env("LLM_REASONER_ALIAS", "malar-reasoner"))
    alias_fast: str = field(default_factory=lambda: _env("LLM_FAST_ALIAS", "malar-fast"))
    alias_vision: str = field(default_factory=lambda: _env("LLM_VISION_ALIAS", "malar-vision"))
    llm_timeout: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT", "120")))

    # Runtime
    data_dir: Path = field(default_factory=lambda: Path(_env("MALAR_DATA_DIR", "data")))
    review_mode: bool = field(default_factory=lambda: _env("MALAR_REVIEW_MODE", "true").lower() == "true")

    # Theorem-guard / memory hyperparameters
    rho: float = 0.9                 # EMA importance weight, 0<rho<1
    theta_novelty: float = 0.15      # insert only when novelty >= theta (> eps)
    eps: float = 1e-3
    merge_contraction: float = 0.7   # c < 1, contractive merge
    merge_radius: float = 0.12
    dt_safety: float = 0.9           # diffusion uses dt = dt_safety * 2/lambda_max

    # Retrieval fusion weights
    w_phi: float = 1.0
    w_h: float = 1.0
    w_g: float = 1.0
    lam_time: float = 0.1
    mu_omega: float = 0.2

    # Conformal / OOD
    conformal_alpha: float = 0.1
    ood_threshold: float = 0.6

    def artifacts_dir(self) -> Path:
        p = self.data_dir / "artifacts"
        p.mkdir(parents=True, exist_ok=True)
        return p


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings()
    return _SETTINGS
