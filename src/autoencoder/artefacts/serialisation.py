"""Save and load model artefacts -- weights, scaler, thresholds, metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch

from autoencoder.model.architecture import Autoencoder

logger = logging.getLogger(__name__)


def save_artefacts(
    output_dir: str | Path,
    model: Autoencoder,
    scaler,
    thresholds: dict,
    metadata: dict,
    training_errors: np.ndarray,
    sensor_baselines: np.ndarray | None = None,
):
    """Save model artefacts to a directory.

    Files saved:
        model_weights.pt
        scaler.pkl
        thresholds.json
        training_metadata.json
        training_error_distribution.npy
        sensor_baselines.npy (optional — per-sensor mean MSE from test split)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "model_weights.pt")
    joblib.dump(scaler, output_dir / "scaler.pkl")

    with open(output_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    np.save(output_dir / "training_error_distribution.npy", training_errors)

    if sensor_baselines is not None:
        np.save(output_dir / "sensor_baselines.npy", sensor_baselines)

    logger.info("Saved all artefacts to %s", output_dir)


def load_artefacts(
    artefact_dir: str | Path,
) -> tuple[Autoencoder, object, dict, dict, np.ndarray, np.ndarray | None]:
    """Load model artefacts from a directory.

    Returns:
        (model, scaler, thresholds, metadata, training_errors, sensor_baselines)
        sensor_baselines is None if the file doesn't exist (backward compat).
    """
    artefact_dir = Path(artefact_dir)

    with open(artefact_dir / "training_metadata.json") as f:
        metadata = json.load(f)

    with open(artefact_dir / "thresholds.json") as f:
        thresholds = json.load(f)

    scaler = joblib.load(artefact_dir / "scaler.pkl")
    training_errors = np.load(artefact_dir / "training_error_distribution.npy")

    baselines_path = artefact_dir / "sensor_baselines.npy"
    sensor_baselines = np.load(baselines_path) if baselines_path.exists() else None

    model = Autoencoder(
        n_sensors=metadata["n_sensors"],
        latent_dim=metadata["latent_dim"],
        max_hidden_layers=metadata.get("max_hidden_layers", 3),
    )
    model.load_state_dict(torch.load(artefact_dir / "model_weights.pt", weights_only=True))
    model.eval()

    logger.info("Loaded artefacts from %s", artefact_dir)
    return model, scaler, thresholds, metadata, training_errors, sensor_baselines
