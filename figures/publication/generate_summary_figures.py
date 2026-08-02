"""Generate public, aggregate-only figures from the canonical thesis tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["Random Forest", "SANNWR", "GWR", "GNNWR", "OLS"]
COLORS = {
    "Random Forest": "#E69F00",
    "SANNWR": "#009E73",
    "GWR": "#D55E00",
    "GNNWR": "#0072B2",
    "OLS": "#7A7A7A",
}


def _ordered(path: str) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / path).set_index("model")
    return frame.loc[MODEL_ORDER]


def validation_comparison() -> None:
    conjunto_prueba = _ordered("conjunto_prueba.csv")
    validacion_aleatoria = _ordered("validacion_cruzada_aleatoria.csv")
    bloques_espaciales = _ordered("validacion_bloques_espaciales.csv")

    x = np.arange(len(MODEL_ORDER))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6.8))

    bars = [
        ax.bar(x - width, conjunto_prueba["rmse"], width, label="Conjunto de prueba 20 %", color="#56B4E9"),
        ax.bar(
            x,
            validacion_aleatoria["rmse_mean"],
            width,
            yerr=validacion_aleatoria["rmse_sd"],
            capsize=3,
            label="Validación cruzada aleatoria",
            color="#0072B2",
        ),
        ax.bar(
            x + width,
            bloques_espaciales["rmse_mean"],
            width,
            yerr=bloques_espaciales["rmse_sd_regions"],
            capsize=3,
            label="Validación por bloques espaciales",
            color="#D55E00",
            alpha=0.88,
        ),
    ]

    for group in bars:
        ax.bar_label(group, fmt="%.1f", padding=3, fontsize=9)

    ax.set_title("El modelo seleccionado cambia con el esquema de validación", weight="bold", pad=14)
    ax.set_ylabel("RMSE (USD/m²)")
    ax.set_xticks(x, MODEL_ORDER)
    ax.set_ylim(0, 225)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.text(
        0.5,
        0.015,
        "Conjunto de prueba: una partición fija. Validación cruzada aleatoria: media ± DE de 5 particiones. "
        "Validación por bloques espaciales: media ± DE de 5 regiones.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT / "validation_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def stability() -> None:
    frame = pd.read_csv(RESULTS / "estabilidad_conjunto_prueba.csv")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = [COLORS[name] for name in frame["model"]]
    bars = ax.barh(frame["model"], frame["rmse_mean"], xerr=frame["rmse_sd"], color=colors, capsize=4)
    ax.bar_label(bars, labels=[f"{m:.2f} ± {s:.2f}" for m, s in zip(frame["rmse_mean"], frame["rmse_sd"])], padding=5)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("RMSE en el conjunto de prueba (USD/m²)")
    ax.set_title("Estabilidad entre diez semillas", weight="bold", pad=12)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "estabilidad_conjunto_prueba.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    validation_comparison()
    stability()
