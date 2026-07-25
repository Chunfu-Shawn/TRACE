"""Plot transcript-level observed and predicted P-site profiles."""

import os
import pickle

import numpy as np
import pandas as pd
import torch
from plotnine import *
from scipy.stats import gaussian_kde, pearsonr
from sklearn.metrics import mean_squared_error


def calculate_density(x, y):
    """Estimate point density for a two-dimensional scatter plot."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 10 or np.ptp(x_clean) == 0 or np.ptp(y_clean) == 0:
        return np.zeros_like(x, dtype=float)

    try:
        density = gaussian_kde(np.vstack([x_clean, y_clean]))(
            np.vstack([x_clean, y_clean])
        )
        density_full = np.zeros_like(x, dtype=float)
        density_full[mask] = density
        return density_full
    except (ValueError, np.linalg.LinAlgError):
        return np.zeros_like(x, dtype=float)


def _to_1d_signal(signal):
    if isinstance(signal, torch.Tensor):
        values = signal.detach().cpu().numpy()
    else:
        values = np.asarray(signal)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        return values
    return values.reshape(values.shape[0], -1).sum(axis=1)


class PredictionVisualizer:
    def __init__(self, pkl_path, dataset, out_dir="./results/plots"):
        """Load predictions and build efficient dataset sample indexes."""
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

        print(f"Loading predictions from {pkl_path}...")
        with open(pkl_path, "rb") as handle:
            self.preds_data = pickle.load(handle)
        if not isinstance(self.preds_data, dict):
            raise ValueError("Prediction pickle must contain a dictionary.")

        self.dataset = dataset
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        cell_count = len(self.preds_data)
        tid_count = sum(
            len(tids) for tids in self.preds_data.values() if isinstance(tids, dict)
        )
        print(
            f"Loaded {tid_count} predictions across {cell_count} cell types. "
            f"Output dir: {self.out_dir}"
        )

        print("Building dataset UUID index...")
        if hasattr(dataset, "uuids"):
            self.uuid_to_idx = {
                str(uuid): index for index, uuid in enumerate(dataset.uuids)
            }
        else:
            self.uuid_to_idx = {
                str(dataset[index][0]): index for index in range(len(dataset))
            }

        self.sample_key_to_idx = {}
        if hasattr(dataset, "cell_types"):
            for index, (uuid, cell_type) in enumerate(
                zip(dataset.uuids, dataset.cell_types)
            ):
                tid = str(uuid).split("-", 1)[0]
                self.sample_key_to_idx.setdefault((tid, str(cell_type)), index)
        print("Index built.")

    def _prediction_for(self, tid, cell_type):
        cell_predictions = self.preds_data.get(cell_type)
        if not isinstance(cell_predictions, dict):
            return None
        if tid in cell_predictions:
            return cell_predictions[tid]
        clean_tid = tid.split(".", 1)[0]
        if clean_tid in cell_predictions:
            return cell_predictions[clean_tid]
        for candidate_tid, prediction in cell_predictions.items():
            if str(candidate_tid).split(".", 1)[0] == clean_tid:
                return prediction
        return None

    def _dataset_index_for(self, tid, cell_type):
        direct = self.sample_key_to_idx.get((tid, cell_type))
        if direct is not None:
            return direct
        clean_tid = tid.split(".", 1)[0]
        for (candidate_tid, candidate_cell), index in self.sample_key_to_idx.items():
            if candidate_cell == cell_type and candidate_tid.split(".", 1)[0] == clean_tid:
                return index
        prefix = f"{tid}-{cell_type}-"
        for uuid, index in self.uuid_to_idx.items():
            if uuid.startswith(prefix):
                return index
        return None

    def plot_transcript(
        self, tid, cell_type, suffix="", ylim: dict = None, log_y: bool = False
    ):
        """Plot one transcript profile and its observation/prediction scatter plot."""
        prediction = self._prediction_for(tid, cell_type)
        if prediction is None:
            print(
                f"Error: Prediction for TID '{tid}' and Cell Type '{cell_type}' "
                "not found in pkl."
            )
            return
        dataset_idx = self._dataset_index_for(tid, cell_type)
        if dataset_idx is None:
            print(
                f"Error: Cannot find TID '{tid}' and Cell Type '{cell_type}' in dataset."
            )
            return

        sample = self.dataset[dataset_idx]
        uuid = str(sample[0])
        print(f"--- Evaluating {uuid} (TID: {tid}, Cell: {cell_type}) ---")
        prediction = _to_1d_signal(prediction)
        truth = _to_1d_signal(sample[6])
        min_len = min(len(truth), len(prediction))
        if min_len < 2:
            print(f"Error: Too few aligned positions for {uuid}.")
            return
        truth = truth[:min_len]
        prediction = prediction[:min_len]

        meta = sample[4]
        cds_start = int(meta.get("cds_start_pos", -1))
        cds_end = int(meta.get("cds_end_pos", -1))
        cds_info = (
            {"start": cds_start, "end": min(cds_end + 3, min_len)}
            if cds_start >= 1 and cds_end >= cds_start
            else None
        )

        valid_mask = np.isfinite(truth) & np.isfinite(prediction)
        truth_valid = truth[valid_mask]
        prediction_valid = prediction[valid_mask]
        if (
            len(truth_valid) > 1
            and np.ptp(truth_valid) > 0
            and np.ptp(prediction_valid) > 0
        ):
            correlation = pearsonr(truth_valid, prediction_valid)
            pcc, p_val = float(correlation.statistic), float(correlation.pvalue)
            mse = float(mean_squared_error(truth_valid, prediction_valid))
        else:
            pcc, p_val = np.nan, np.nan
            mse = float(mean_squared_error(truth_valid, prediction_valid)) if len(truth_valid) else np.nan

        bar_plot_name = f"{uuid}_psite.{suffix}.pdf"
        self._plot_psite_density_bar_plotnine(
            truth=truth,
            pred=prediction,
            cds_info=cds_info,
            save_path=os.path.join(self.out_dir, bar_plot_name),
            pcc=pcc,
            p_val=p_val,
            ylim=ylim,
            log_y=log_y,
        )

        scatter_plot_name = f"{uuid}_scatter.{suffix}.pdf"
        self._plot_correlation_scatter_plotnine(
            truth=truth,
            pred=prediction,
            pcc=pcc,
            p_val=p_val,
            mse=mse,
            save_path=os.path.join(self.out_dir, scatter_plot_name),
        )

    def _plot_psite_density_bar_plotnine(
        self,
        truth,
        pred,
        cds_info,
        save_path,
        pcc=None,
        p_val=None,
        ylim=None,
        log_y=False,
    ):
        """Plot observed and predicted profiles in vertically stacked facets."""
        del p_val
        x = np.arange(len(truth))
        cds_start_idx = cds_info["start"] - 1 if cds_info else 0
        frames = (x - cds_start_idx) % 3
        df_truth = pd.DataFrame(
            {"Pos": x, "Density": truth, "Frame": frames, "Source": "Observation"}
        )
        df_pred = pd.DataFrame(
            {"Pos": x, "Density": pred, "Frame": frames, "Source": "Prediction"}
        )
        df = pd.concat([df_truth, df_pred], ignore_index=True)
        source_categories = ["Observation", "Prediction"]
        df["Frame"] = df["Frame"].astype(str)
        df["Source"] = pd.Categorical(df["Source"], categories=source_categories)

        if isinstance(ylim, dict):
            for source in source_categories:
                if source in ylim:
                    y_min, y_max = ylim[source]
                    mask = df["Source"] == source
                    df.loc[mask, "Density"] = df.loc[mask, "Density"].clip(
                        lower=y_min, upper=y_max
                    )

        soft_colors = {"0": "#D73027", "1": "#4575B4", "2": "darkgray"}
        plot = ggplot(df, aes(x="Pos", y="Density", fill="Frame"))
        if cds_info:
            rect_df = pd.DataFrame(
                {
                    "xmin": [cds_info["start"] - 1],
                    "xmax": [cds_info["end"]],
                    "ymin": [-np.inf],
                    "ymax": [np.inf],
                }
            )
            plot += geom_rect(
                data=rect_df,
                mapping=aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                alpha=0.1,
                fill="gray",
                inherit_aes=False,
            )

        if isinstance(ylim, dict):
            blank_data = []
            for source in source_categories:
                if source in ylim:
                    y_min, y_max = ylim[source]
                    blank_data.extend(
                        [
                            {"Source": source, "Pos": 0, "Density": y_min},
                            {"Source": source, "Pos": 0, "Density": y_max},
                        ]
                    )
            if blank_data:
                blank_df = pd.DataFrame(blank_data)
                blank_df["Source"] = pd.Categorical(
                    blank_df["Source"], categories=source_categories
                )
                plot += geom_blank(
                    data=blank_df,
                    mapping=aes(x="Pos", y="Density"),
                    inherit_aes=False,
                )

        plot = (
            plot
            + geom_col(width=1.0, size=0)
            + facet_wrap("~Source", ncol=1, scales="free_y")
            + scale_fill_manual(
                values=soft_colors, labels=["Frame 0", "Frame 1", "Frame 2"]
            )
            + scale_x_continuous(expand=(0, 0), limits=(-0.5, len(truth) - 0.5))
        )

        if pcc is not None:
            annot_text = f"R = {pcc:.2f}" if np.isfinite(pcc) else "R = NA"
            max_x = df["Pos"].max()
            if isinstance(ylim, dict) and "Prediction" in ylim:
                text_y_pos = ylim["Prediction"][1] * 0.85
            else:
                text_y_pos = max(float(df_pred["Density"].max()) * 0.85, 1e-8)
            annot_df = pd.DataFrame(
                {
                    "Pos": [max_x * 0.95],
                    "Density": [text_y_pos],
                    "Source": ["Prediction"],
                    "Label": [annot_text],
                }
            )
            annot_df["Source"] = pd.Categorical(
                annot_df["Source"], categories=source_categories
            )
            plot += geom_text(
                data=annot_df,
                mapping=aes(x="Pos", y="Density", label="Label"),
                ha="right",
                va="top",
                size=14,
                fontstyle="italic",
                color="black",
                inherit_aes=False,
            )

        y_axis_label = "Translation signal"
        if log_y:
            y_axis_label = "Translation signal (log1p)"
            plot += scale_y_continuous(trans="log1p")

        plot = (
            plot
            + theme_classic()
            + theme(
                figure_size=(10, 4),
                legend_position="top",
                legend_direction="horizontal",
                legend_title=element_text(size=14, color="black"),
                legend_text=element_text(size=12, color="black"),
                strip_background=element_blank(),
                strip_text=element_text(size=16, color="black"),
                axis_line_y=element_blank(),
                axis_ticks_major_y=element_line(color="black", size=0.8),
                axis_line_x=element_line(color="black", size=0.8),
                axis_ticks_major_x=element_line(color="black", size=0.8),
                panel_grid_major=element_line(
                    color="#E0E0E0", size=0.8, alpha=0.8
                ),
                panel_grid_minor=element_blank(),
                axis_title=element_text(size=14, color="black"),
                axis_text=element_text(size=12, color="black"),
            )
            + labs(
                x="Transcript Position (nt)",
                y=y_axis_label,
                fill="Reading frame",
            )
        )
        plot.save(save_path, verbose=False)
        print(f"Periodicity plot saved: {save_path}")

    def _plot_correlation_scatter_plotnine(
        self, truth, pred, pcc, p_val, mse, save_path
    ):
        """Plot observation/prediction scatter points colored by local density."""
        del p_val
        df = pd.DataFrame({"True": truth, "Predicted": pred})
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if df.empty:
            print(f"No finite points available for scatter plot: {save_path}")
            return
        df["density"] = calculate_density(
            df["True"].to_numpy(), df["Predicted"].to_numpy()
        )
        df = df.sort_values("density")
        pcc_label = f"{pcc:.3f}" if np.isfinite(pcc) else "NA"
        mse_label = f"{mse:.3f}" if np.isfinite(mse) else "NA"
        plot = (
            ggplot(df, aes(x="True", y="Predicted", color="density"))
            + geom_point(alpha=0.7, size=1.5, stroke=0, show_legend=False)
            + scale_color_cmap(cmap_name="magma")
            + theme_classic()
            + theme(
                figure_size=(5, 5),
                axis_ticks_major_y=element_blank(),
                panel_border=element_rect(color="black", fill=None, size=1),
            )
            + labs(
                title=f"Correlation: R={pcc_label}, MSE={mse_label}",
                x="Observation",
                y="Prediction",
            )
        )
        plot.save(save_path, verbose=False)
        print(f"Scatter plot saved: {save_path}")
