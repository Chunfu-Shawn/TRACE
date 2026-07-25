"""TIS/TTS metagene evaluation for positional translation predictions."""

import os
import pickle

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_text,
    facet_wrap,
    geom_line,
    geom_rect,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_identity,
    scale_linetype_manual,
    theme,
    theme_bw,
)
from tqdm import tqdm

try:
    from .evaluation_utils import cds_slice, get_prediction, to_1d_signal, transcript_id_from_uuid
except ImportError:
    from evaluation_utils import cds_slice, get_prediction, to_1d_signal, transcript_id_from_uuid


def evaluate_metagene_TIS_TTS_profile(
    dataset,
    pkl_path,
    window_size=12,
    out_dir="./results/plots",
    suffix="",
    unlog_data=True,
):
    """Plot aggregate density around translation start and stop codons.

    Parameters
    ----------
    dataset
        A TranslationDataset-compatible object containing targets and metadata.
    pkl_path
        Prediction PKL with ``{cell_type: {transcript_id: signal}}`` structure.
    window_size
        Number of nucleotides retained on each side of the landmark.
    out_dir
        Output directory for the PDF figure.
    suffix
        Optional output filename suffix.
    unlog_data
        Apply ``expm1`` before aggregation. This is recommended for log1p targets.

    Returns
    -------
    None
        The function preserves its historical plotting-only return contract.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError("window_size must be a positive integer")

    print(f"Loading predictions from {pkl_path}...")
    with open(pkl_path, "rb") as handle:
        preds_data = pickle.load(handle)
    if not isinstance(preds_data, dict):
        raise ValueError("Prediction pickle must contain a dictionary.")

    os.makedirs(out_dir, exist_ok=True)
    vector_length = 2 * window_size
    sums = {
        "tis_gt": np.zeros(vector_length, dtype=np.float64),
        "tis_pred": np.zeros(vector_length, dtype=np.float64),
        "tts_gt": np.zeros(vector_length, dtype=np.float64),
        "tts_pred": np.zeros(vector_length, dtype=np.float64),
    }

    valid_count = 0
    print(f"Scanning {len(dataset)} transcripts in dataset...")
    for index in tqdm(range(len(dataset))):
        sample = dataset[index]
        uuid = str(sample[0])
        tid = transcript_id_from_uuid(uuid)
        cell_type = str(sample[2])
        prediction = get_prediction(preds_data, cell_type, tid)
        if prediction is None:
            continue

        truth = to_1d_signal(sample[6])
        length = min(len(truth), len(prediction))
        if length < vector_length:
            continue
        truth = truth[:length]
        prediction = prediction[:length].astype(np.float32, copy=False)

        if unlog_data:
            truth = np.expm1(truth)
            prediction = np.expm1(prediction)

        bounds = cds_slice(sample[4], length)
        if bounds is None:
            continue
        start_idx, end_idx = bounds
        stop_codon_start = end_idx
        if stop_codon_start + 3 > length:
            continue

        tis_start = start_idx - window_size
        tis_end = start_idx + window_size
        tts_start = stop_codon_start - window_size
        tts_end = stop_codon_start + window_size
        if tis_start < 0 or tts_start < 0 or tis_end > length or tts_end > length:
            continue

        sums["tis_gt"] += truth[tis_start:tis_end]
        sums["tis_pred"] += prediction[tis_start:tis_end]
        sums["tts_gt"] += truth[tts_start:tts_end]
        sums["tts_pred"] += prediction[tts_start:tts_end]
        valid_count += 1

    print(f"Used {valid_count} valid transcripts for meta-gene analysis.")
    if valid_count == 0:
        print("No valid transcripts found.")
        return

    eps = 1e-6
    norm_tis_gt = sums["tis_gt"] / (sums["tis_gt"].sum() + eps)
    norm_tis_pred = sums["tis_pred"] / (sums["tis_pred"].sum() + eps)
    norm_tts_gt = sums["tts_gt"] / (sums["tts_gt"].sum() + eps)
    norm_tts_pred = sums["tts_pred"] / (sums["tts_pred"].sum() + eps)

    positions = np.arange(-window_size, window_size)
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "Position": positions,
                    "Density": norm_tis_gt,
                    "Source": "Ground Truth",
                    "Region": "TIS (Start)",
                }
            ),
            pd.DataFrame(
                {
                    "Position": positions,
                    "Density": norm_tis_pred,
                    "Source": "Prediction",
                    "Region": "TIS (Start)",
                }
            ),
            pd.DataFrame(
                {
                    "Position": positions,
                    "Density": norm_tts_gt,
                    "Source": "Ground Truth",
                    "Region": "TTS (Stop)",
                }
            ),
            pd.DataFrame(
                {
                    "Position": positions,
                    "Density": norm_tts_pred,
                    "Source": "Prediction",
                    "Region": "TTS (Stop)",
                }
            ),
        ],
        ignore_index=True,
    )
    df["Source"] = pd.Categorical(
        df["Source"], categories=["Ground Truth", "Prediction"]
    )

    annot_cds = pd.DataFrame(
        [
            {"Region": "TIS (Start)", "xmin": 0, "xmax": window_size},
            {"Region": "TTS (Stop)", "xmin": -window_size, "xmax": 3},
        ]
    )
    annot_codon = pd.DataFrame(
        [
            {
                "Region": "TIS (Start)",
                "xmin": 0,
                "xmax": 3,
                "color": "limegreen",
            },
            {
                "Region": "TTS (Stop)",
                "xmin": 0,
                "xmax": 3,
                "color": "#e74c3c",
            },
        ]
    )

    plot = (
        ggplot(df, aes(x="Position", y="Density"))
        + geom_rect(
            data=annot_cds,
            mapping=aes(xmin="xmin", xmax="xmax", ymin=-np.inf, ymax=np.inf),
            fill="gray",
            alpha=0.15,
            inherit_aes=False,
        )
        + geom_rect(
            data=annot_codon,
            mapping=aes(
                xmin="xmin", xmax="xmax", ymin=-np.inf, ymax=np.inf, fill="color"
            ),
            alpha=0.2,
            inherit_aes=False,
            show_legend=False,
        )
        + scale_fill_identity()
        + geom_line(aes(color="Source", linetype="Source"), size=1)
        + scale_color_manual(
            values={"Ground Truth": "darkgray", "Prediction": "#005b96"}
        )
        + scale_linetype_manual(
            values={"Ground Truth": "solid", "Prediction": "dashed"}
        )
        + facet_wrap("~Region", ncol=2, scales="free_y")
        + theme_bw()
        + theme(
            strip_background=element_blank(),
            strip_text=element_text(size=11),
            legend_title=element_blank(),
            legend_text=element_text(size=11),
            legend_position="top",
        )
        + labs(
            x="Distance from Start/Stop Codon (nt)",
            y="Normalized P-site Density",
        )
    )

    file_suffix = f".{suffix}" if suffix else ""
    save_file = os.path.join(out_dir, f"metagene_tis_tts_profile{file_suffix}.pdf")
    plot.save(save_file, width=8, height=4, verbose=False)
    print(f"Metagene plot saved to: {save_file}")
