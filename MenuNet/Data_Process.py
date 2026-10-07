import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from scipy import stats
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
import ast
    
class data_processor():
    def __init__(self, csv_name):
         self.csv_name = csv_name
         self.csv_path = "MenuNet/Auction_Results/" + self.csv_name

    def store_data(self, summary):
        folder_path = os.path.join(os.path.dirname(__file__), "Auction_Results")
        os.makedirs(folder_path, exist_ok=True)
        df = pd.DataFrame([summary])
        file_exists = os.path.isfile(self.csv_path)
        df.to_csv(self.csv_path, mode="a", header = not file_exists, index = False)

    def menu_option_means(self, category):
        df = pd.read_csv(self.csv_path)
        df1 = df.loc[df[category], ["aggregate_menu_counts"]]
        df2 = df.loc[df[category]==False,["aggregate_menu_counts"]]
        df1_expanded = pd.DataFrame([ast.literal_eval(x) for x in df1["aggregate_menu_counts"]])
        df2_expanded = pd.DataFrame([ast.literal_eval(x) for x in df2["aggregate_menu_counts"]])
        print(df1_expanded.mean(), "\n\n", df2_expanded.mean(), "\n")
        for option in df1_expanded.columns:
            t_stat, p_val = stats.ttest_ind(df1_expanded[option], df2_expanded[option])
            print(f"{option}:")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_val:.4f}")


    def compare_auctions(
        self,
        v1,
        v2,
        v3: bool = False,
        *,
        save_dir: str | None = None,
        save_prefix: str = "",
        show: bool = True,
    ):
        # Compare across multiple result CSVs (including any *_v2.csv present).
        # When comparing v1 vs v2 together, keep plots "history only" (clean A/B).
        # Otherwise (only v1 or only v2 selected), include both history and no-history runs.
        base_conditions = ("rule", "same_bidder")
        plot_conditions = base_conditions if (v1 and v2) else ("rule", "hist", "same_bidder")
        metrics = ("avg_truth", "avg_regret")

        existing_figs = set(plt.get_fignums())
        
        plt.style.use("ggplot")
        plt.rcParams.update(
                {"axes.prop_cycle": cycler(color=["#87CEEB", "#4C78A8", "#72B7B2", "#FC5656",  "#911010", "#E4551C",]),
                "axes.facecolor": "#F3FAFD",
                "figure.facecolor": "#FFFFFF",})
        
        base_dir = os.path.join(os.path.dirname(__file__), "Auction_Results")
        paths = {}
        reasoning_pairs = {}
        other_paths = {}

        if v1:
            v1_min = os.path.join(base_dir, "GPT-5-mini_Min.csv")
            v1_low = os.path.join(base_dir, "GPT-5-mini_Low.csv")
            v1_med = os.path.join(base_dir, "GPT-5-mini_Med.csv")
            paths["GPT-5-mini Min"] = v1_min
            paths["GPT-5-mini Low"] = v1_low
            paths["GPT-5-mini Med"] = v1_med
            reasoning_pairs.setdefault("Min", {})["v1"] = v1_min
            reasoning_pairs.setdefault("Low", {})["v1"] = v1_low
            reasoning_pairs.setdefault("Med", {})["v1"] = v1_med
        if v2:
            v2_min = os.path.join(base_dir, "GPT-5-mini_Min_v2.csv")
            v2_low = os.path.join(base_dir, "GPT-5-mini_Low_v2.csv")
            v2_med = os.path.join(base_dir, "GPT-5-mini_Med_v2.csv")
            paths["GPT-5-mini Min v2"] = v2_min
            paths["GPT-5-mini Low v2"] = v2_low
            paths["GPT-5-mini Med v2"] = v2_med
            reasoning_pairs.setdefault("Min", {})["v2"] = v2_min
            reasoning_pairs.setdefault("Low", {})["v2"] = v2_low
            reasoning_pairs.setdefault("Med", {})["v2"] = v2_med

        if v3:
            # Compare GPT-4o and all GPT-5.2 CSVs (history-only).
            gpt4o_path = os.path.join(base_dir, "GPT-4o.csv")
            if os.path.exists(gpt4o_path):
                other_paths["GPT-4o"] = gpt4o_path

            for entry in sorted(os.listdir(base_dir)):
                if entry.startswith("GPT-5.2_") and entry.endswith(".csv"):
                    label = entry.replace(".csv", "").replace("GPT-5.2_", "GPT-5.2 ")
                    other_paths[label] = os.path.join(base_dir, entry)

        frames = []
        for label, path in paths.items():
            df = pd.read_csv(path)
            if (v1 and v2) and ("hist" in df.columns):
                df = df.loc[df["hist"] == True].copy()
            df = df[list(plot_conditions) + list(metrics)].copy()
            df["source"] = label
            frames.append(df)

        if frames:
            all_df = pd.concat(frames, ignore_index=True)
            grouped = (
                all_df.groupby(list(plot_conditions) + ["source"], dropna=False)[list(metrics)].mean().reset_index()
            )
            grouped["condition_label"] = grouped[list(plot_conditions)].astype(str).agg("|".join, axis=1)
            min_avg_truth = float(0)
            for label, subset in grouped.groupby("condition_label", sort=False):
                print(f"\nCondition {label}")
                for metric in metrics:
                    avg = subset.groupby("source", sort=False)[metric].mean()
                    print(f"  {metric}:")
                    for source, value in avg.items():
                        print(f"    {source}: {value:.4f}")

            source_order = list(paths.keys())
            fig, axes = plt.subplots(2, 1, figsize=(9, 10))
            for ax, metric in zip(axes, metrics):
                pivot = grouped.pivot_table(
                    index="condition_label",
                    columns="source",
                    values=metric,
                    aggfunc="mean",
                )
                pivot = pivot.reindex(columns=source_order)
                pivot.plot(kind="bar", ax=ax, legend=False)
                if v1 and v2:
                    ax.set_title("Average by Auction Preset", fontsize=16)
                else:
                    title_metric = "Truth" if metric == "avg_truth" else "Regret"
                    ax.set_title(f"Average {title_metric} by Auction Preset", fontsize=16)
                ax.set_xlabel("|".join(plot_conditions), fontdict={"size": 12})
                ax.set_ylabel(metric, fontdict={"size": 12})
                ax.tick_params(axis="x", labelsize=10)
                ax.tick_params(axis="y", labelsize=10)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="center")
                ax.set_axisbelow(True)
                # Keep only horizontal (y) grid lines and ensure no vertical grid lines show through bars.
                ax.grid(False, axis="x", which="both")
                ax.grid(True, axis="y", which="major", color="#D0D7DE", linewidth=0.8, alpha=0.9, zorder=0)
                ax.grid(False, axis="y", which="minor")
                ax.minorticks_off()
                if metric == "avg_truth":
                    ax.set_ylim(bottom=15)
                elif metric == "avg_regret":
                    ax.set_ylim(-0.25, 0.0)
                    # Indicate when v2 bars are clipped by the y-limits.
                    y_min, y_max = ax.get_ylim()
                    for container in ax.containers:
                        label = getattr(container, "get_label", lambda: "")()
                        if "v2" not in str(label):
                            continue
                        for patch in getattr(container, "patches", []):
                            value = patch.get_y() + patch.get_height()
                            x_center = patch.get_x() + patch.get_width() / 2
                            face = patch.get_facecolor()
                            if value < y_min:
                                ax.plot(
                                    x_center,
                                    y_min,
                                    marker="v",
                                    markersize=7,
                                    color=face,
                                    markeredgecolor="none",
                                    clip_on=False,
                                    zorder=10,
                                )
                            elif value > y_max:
                                ax.plot(
                                    x_center,
                                    y_max,
                                    marker="^",
                                    markersize=7,
                                    color=face,
                                    markeredgecolor="none",
                                    clip_on=False,
                                    zorder=10,
                                )

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, title="CSV", fontsize=10, title_fontsize=10, loc="upper left")
            fig.tight_layout()

        if v1 and v2:
            # 4 x-ticks for auction presets; per tick: 3 bars (Min/Low/Med).
            # For each bar, draw v1 (lighter shade) and v2 (darker shade) at the same x-position.
            # The shorter bar is drawn in front so the visible cap shows the delta.
            reasoning_order = ["Min", "Low", "Med"]
            overlay_frames = []

            for reasoning_label in reasoning_order:
                pair = reasoning_pairs.get(reasoning_label, {})
                v1_path = pair.get("v1")
                v2_path = pair.get("v2")
                if not v1_path or not v2_path:
                    continue
                if not (os.path.exists(v1_path) and os.path.exists(v2_path)):
                    continue

                v1_df = pd.read_csv(v1_path)
                if "hist" in v1_df.columns:
                    v1_df = v1_df.loc[v1_df["hist"] == True].copy()
                v1_df = v1_df[list(base_conditions) + list(metrics)].copy()
                v1_df["version"] = "v1"
                v1_df["reasoning"] = reasoning_label

                v2_df = pd.read_csv(v2_path)
                if "hist" in v2_df.columns:
                    v2_df = v2_df.loc[v2_df["hist"] == True].copy()
                v2_df = v2_df[list(base_conditions) + list(metrics)].copy()
                v2_df["version"] = "v2"
                v2_df["reasoning"] = reasoning_label

                overlay_frames.append(v1_df)
                overlay_frames.append(v2_df)

            if overlay_frames:
                overlay_df = pd.concat(overlay_frames, ignore_index=True)
                overlay_grouped = (
                    overlay_df.groupby(list(base_conditions) + ["reasoning", "version"], dropna=False)[list(metrics)]
                    .mean()
                    .reset_index()
                )

                presets = [
                    (False, False),
                    (False, True),
                    (True, False),
                    (True, True),
                ]
                preset_labels = [
                    "No rule\nDiff bidders",
                    "No rule\nSame bidder",
                    "Rule\nDiff bidders",
                    "Rule\nSame bidder",
                ]

                # User-specified coloring: v1 uses one light blue for all reasoning levels,
                # v2 uses one darker blue for all reasoning levels.
                colors_v1 = {
                    "Min": "#87CEEB",
                    "Low": "#87CEEB",
                    "Med": "#87CEEB",
                }
                colors_v2 = {
                    "Min": "#4C78A8",
                    "Low": "#4C78A8",
                    "Med": "#4C78A8",
                }

                group_x = np.arange(len(presets), dtype=float)
                group_width = 0.78
                bar_width = (group_width / len(reasoning_order)) * 0.92
                offsets = np.linspace(
                    -group_width / 2 + bar_width / 2,
                    group_width / 2 - bar_width / 2,
                    num=len(reasoning_order),
                )

                for metric in metrics:
                    metric_pivot = overlay_grouped.pivot_table(
                        index=list(base_conditions) + ["reasoning"],
                        columns="version",
                        values=metric,
                        aggfunc="mean",
                    ).reindex(columns=["v1", "v2"])

                    legend_handles = [
                        Patch(facecolor=colors_v1["Min"], edgecolor="none", label="Prompt 1"),
                        Patch(facecolor=colors_v2["Min"], edgecolor="none", label="Prompt 2"),
                        Line2D(
                            [0],
                            [0],
                            marker="v",
                            color="none",
                            markerfacecolor="#4C78A8",
                            markersize=7,
                            label="Prompt 2 clipped",
                        ),
                    ]

                    def _plot_on_axis(ax, *, y_min=None, y_max=None):
                        for r_i, reasoning_label in enumerate(reasoning_order):
                            positions = group_x + offsets[r_i] #type: ignore
                            v1_vals = []
                            v2_vals = []
                            for rule_val, same_val in presets:
                                key = (rule_val, same_val, reasoning_label)
                                if key in metric_pivot.index:
                                    v1_vals.append(metric_pivot.loc[key, "v1"])
                                    v2_vals.append(metric_pivot.loc[key, "v2"])
                                else:
                                    v1_vals.append(np.nan)
                                    v2_vals.append(np.nan)

                            v1_color = colors_v1.get(reasoning_label, "#87CEEB")
                            v2_color = colors_v2.get(reasoning_label, "#4C78A8")

                            for x, h1, h2 in zip(positions, v1_vals, v2_vals):
                                h1_nan = pd.isna(h1)
                                h2_nan = pd.isna(h2)
                                if h1_nan and h2_nan:
                                    continue
                                if h2_nan:
                                    ax.bar(x, h1, width=bar_width, color=v1_color, edgecolor="none", zorder=2)
                                    continue
                                if h1_nan:
                                    ax.bar(x, h2, width=bar_width, color=v2_color, edgecolor="none", zorder=2)
                                    continue

                                # Overlay bars at the same x so we see both v1 and v2 with fixed colors.
                                # The shorter-magnitude bar (closer to 0) is drawn in front.
                                v1_abs = abs(float(h1))
                                v2_abs = abs(float(h2))
                                if v1_abs <= v2_abs:
                                    outer_h, outer_c, outer_z = h2, v2_color, 3
                                    inner_h, inner_c, inner_z = h1, v1_color, 4
                                else:
                                    outer_h, outer_c, outer_z = h1, v1_color, 3
                                    inner_h, inner_c, inner_z = h2, v2_color, 4

                                ax.bar(x, outer_h, width=bar_width, color=outer_c, edgecolor="none", zorder=outer_z)
                                ax.bar(x, inner_h, width=bar_width, color=inner_c, edgecolor="none", zorder=inner_z)

                                if y_min is not None and not pd.isna(h2):
                                    if h2 < y_min:
                                        ax.plot(
                                            x,
                                            y_min,
                                            marker="v",
                                            markersize=7,
                                            color=v2_color,
                                            markeredgecolor="none",
                                            clip_on=False,
                                            zorder=10,
                                        )
                                    elif y_max is not None and h2 > y_max:
                                        ax.plot(
                                            x,
                                            y_max,
                                            marker="^",
                                            markersize=7,
                                            color=v2_color,
                                            markeredgecolor="none",
                                            clip_on=False,
                                            zorder=10,
                                        )

                        ax.tick_params(axis="y", labelsize=12)

                    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
                    y_min = y_max = None
                    if metric == "avg_regret":
                        y_min, y_max = -0.4, 0.0
                        ax.set_ylim(y_min, y_max)
                    _plot_on_axis(ax, y_min=y_min, y_max=y_max)
                    if metric == "avg_truth":
                        title = "GPT-5-Mini: Avg History for Prompt 1 vs Prompt 2 sorted by Reasoning"
                    else:
                        title = "GPT-5-Mini: Avg Regret for Prompt 1 vs Prompt 2 sorted by Reasoning"
                    ax.set_title(title, fontsize=18)
                    ax.set_xlabel("Auction preset", fontdict={"size": 14})
                    ax.set_ylabel(metric, fontdict={"size": 14})
                    ax.set_xticks(group_x)
                    ax.set_xticklabels(preset_labels, rotation=0, ha="center", fontsize=12)
                    ax.set_axisbelow(True)
                    # Keep only horizontal (y) grid lines and ensure no vertical grid lines show through bars.
                    ax.grid(False, axis="x", which="both")
                    ax.grid(True, axis="y", which="major", color="#E1E7EE", linewidth=0.8, alpha=1.0, zorder=0)
                    ax.grid(False, axis="y", which="minor")
                    ax.minorticks_off()
                    for spine in ("top", "right", "left", "bottom"):
                        ax.spines[spine].set_visible(False)

                    # Add reasoning labels along the top (per bar/column) since v1/v2 colors are shared
                    # across reasoning levels.
                    top_ax = ax.twiny()
                    top_ax.set_xlim(ax.get_xlim())
                    top_ticks = []
                    top_labels = []
                    for r_i, reasoning_label in enumerate(reasoning_order):
                        for x in (group_x + offsets[r_i]):
                            top_ticks.append(float(x))
                            top_labels.append(reasoning_label)
                    top_ax.set_xticks(top_ticks)
                    top_ax.set_xticklabels(top_labels, rotation=0, ha="center", fontsize=11)
                    top_ax.tick_params(axis="x", pad=2, length=0)
                    # Ensure the twinned axis doesn't draw its own (vertical) grid lines.
                    top_ax.grid(False, axis="both", which="both")
                    top_ax.minorticks_off()
                    top_ax.patch.set_alpha(0)
                    for spine in ("top", "bottom", "left", "right"):
                        top_ax.spines[spine].set_visible(False)
                    if metric == "avg_truth":
                        ax.set_ylim(bottom=15)
                    elif metric == "avg_regret":
                        ax.set_ylim(-0.4, 0.0)
                    ax.legend(handles=legend_handles, loc="lower left", fontsize=10, framealpha=0.9)

                    fig.tight_layout()

        # AB summary plot (Minimal, 30 rounds, 30 runs) if present.
        # Only show this when comparing both prompts (v1 vs v2) together.
        ab_path = os.path.join(base_dir, "AB Min 30r,30runs.csv")
        if (v1 and v2) and os.path.exists(ab_path):
            ab_df = pd.read_csv(ab_path)
            if "prompt_type" in ab_df.columns and "truncate_history" in ab_df.columns:
                ab_df = ab_df.copy()
                ab_df["prompt_display"] = ab_df["prompt_type"].replace({"MenuNet2": "MenuNet old"})
                grouped_ab = (
                    ab_df.groupby(["prompt_display", "truncate_history"], dropna=False)[["truth_rate", "regret"]]
                    .mean()
                    .reset_index()
                )

                # Prompt 1 = MenuNet old (formerly MenuNet2), Prompt 2 = MenuNet.
                prompt_order = ["MenuNet old", "MenuNet"]
                prompt_labels = {"MenuNet old": "Prompt 1", "MenuNet": "Prompt 2"}
                trunc_order = [False, True]
                trunc_labels = {False: "untruncated", True: "truncated"}
                trunc_colors = {False: "#87CEEB", True: "#4C78A8"}

                group_x = np.arange(len(prompt_order), dtype=float)
                bar_width = 0.34
                offsets = {False: -bar_width / 2, True: bar_width / 2}

                fig, (ax_truth, ax_regret) = plt.subplots(1, 2, figsize=(13, 5))
                fig.suptitle(
                    "GPT-5-Mini: Minimal Reasoning, Comparing Prompts Controlling for History Truncation",
                    fontsize=14,
                )

                for trunc in trunc_order:
                    xs = group_x + offsets[trunc]
                    truth_vals = []
                    regret_vals = []
                    for prompt in prompt_order:
                        row = grouped_ab.loc[
                            (grouped_ab["prompt_display"] == prompt)
                            & (grouped_ab["truncate_history"] == trunc)
                        ]
                        if row.empty:
                            truth_vals.append(np.nan)
                            regret_vals.append(np.nan)
                        else:
                            truth_vals.append(float(row["truth_rate"].iloc[0]))
                            regret_vals.append(float(row["regret"].iloc[0]))

                    ax_truth.bar(
                        xs,
                        truth_vals,
                        width=bar_width,
                        color=trunc_colors[trunc],
                        edgecolor="none",
                        label=trunc_labels[trunc],
                    )
                    ax_regret.bar(
                        xs,
                        regret_vals,
                        width=bar_width,
                        color=trunc_colors[trunc],
                        edgecolor="none",
                        label=trunc_labels[trunc],
                    )

                ax_truth.set_title("Mean truth rate", fontsize=16)
                ax_truth.set_ylabel("truth_rate", fontdict={"size": 12})
                ax_truth.set_xticks(group_x)
                ax_truth.set_xticklabels([prompt_labels[p] for p in prompt_order], fontsize=12)
                ax_truth.set_ylim(0.0, 1.0)

                ax_regret.set_title("Mean regret", fontsize=16)
                ax_regret.set_ylabel("regret", fontdict={"size": 12})
                ax_regret.set_xticks(group_x)
                ax_regret.set_xticklabels([prompt_labels[p] for p in prompt_order], fontsize=12)
                ax_regret.axhline(0, color="#666666", linewidth=1.0, alpha=0.7)
                if not grouped_ab["regret"].isna().all():
                    min_regret = float(grouped_ab["regret"].min())
                    ax_regret.set_ylim(min_regret * 1.1, 0.05)

                ax_truth.legend(title="history", fontsize=10, title_fontsize=10, loc="lower left", framealpha=0.9)
                ax_regret.legend(title="history", fontsize=10, title_fontsize=10, loc="lower left", framealpha=0.9)
                fig.tight_layout()

        if v3:
            def _load_sources(source_to_paths: dict[str, list[str]]) -> pd.DataFrame:
                frames_local: list[pd.DataFrame] = []
                for label, paths_list in source_to_paths.items():
                    existing = [p for p in paths_list if p and os.path.exists(p)]
                    if not existing:
                        continue
                    model_frames: list[pd.DataFrame] = []
                    for path in existing:
                        df = pd.read_csv(path)
                        if (v1 and v2) and ("hist" in df.columns):
                            df = df.loc[df["hist"] == True].copy()
                        df = df[list(plot_conditions) + list(metrics)].copy()
                        model_frames.append(df)
                    if not model_frames:
                        continue
                    merged = pd.concat(model_frames, ignore_index=True)
                    merged["source"] = label
                    frames_local.append(merged)
                if not frames_local:
                    return pd.DataFrame()
                return pd.concat(frames_local, ignore_index=True)

            def _plot_two_metric_stack(
                grouped_df: pd.DataFrame,
                source_order: list[str],
                title: str,
                *,
                regret_ylim: tuple[float, float] | None = None,
            ) -> None:
                if grouped_df.empty:
                    return
                fig, axes = plt.subplots(2, 1, figsize=(9, 10))
                for ax, metric in zip(axes, metrics):
                    pivot = grouped_df.pivot_table(
                        index="condition_label",
                        columns="source",
                        values=metric,
                        aggfunc="mean",
                    ).reindex(columns=source_order)
                    pivot.plot(kind="bar", ax=ax, legend=False)
                    if metric == "avg_truth":
                        ax.set_title("Average Truth by Auction Preset", fontsize=16)
                    else:
                        ax.set_title("Average Regret by Auction Preset", fontsize=16)
                    ax.set_xlabel("|".join(plot_conditions), fontdict={"size": 12})
                    ax.set_ylabel(metric, fontdict={"size": 12})
                    ax.tick_params(axis="x", labelsize=10)
                    ax.tick_params(axis="y", labelsize=10)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="center")
                    ax.set_axisbelow(True)
                    ax.grid(False, axis="x", which="both")
                    ax.grid(True, axis="y", which="major", color="#D0D7DE", linewidth=0.8, alpha=0.9, zorder=0)
                    ax.grid(False, axis="y", which="minor")
                    ax.minorticks_off()

                    # For v3 plots, auto-scale y so all values fit.
                    vals = pivot.to_numpy().astype(float).ravel()
                    vals = vals[~np.isnan(vals)]
                    if metric == "avg_regret" and regret_ylim is not None:
                        ax.set_ylim(regret_ylim[0], regret_ylim[1])
                    elif vals.size:
                        vmin = float(vals.min())
                        vmax = float(vals.max())
                        if vmin == vmax:
                            pad = 1.0 if metric == "avg_truth" else 0.1
                        else:
                            pad = (vmax - vmin) * 0.08
                        if metric == "avg_truth":
                            ax.set_ylim(max(0.0, vmin - pad), vmax + pad)
                        else:
                            ax.set_ylim(vmin - pad, vmax + pad)

                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper left", fontsize=10, framealpha=0.9)
                fig.tight_layout()

            # Graph 1: GPT-4o vs GPT-5.2 None vs GPT-5-mini (all reasoning) vs GPT-5-mini v2 (all reasoning).
            gpt4o_path = os.path.join(base_dir, "GPT-4o.csv")
            gpt52_none_path = os.path.join(base_dir, "GPT-5.2_None.csv")
            gpt5mini_v1_paths = [
                os.path.join(base_dir, "GPT-5-mini_Min.csv"),
                os.path.join(base_dir, "GPT-5-mini_Low.csv"),
                os.path.join(base_dir, "GPT-5-mini_Med.csv"),
            ]
            gpt5mini_v2_paths = [
                os.path.join(base_dir, "GPT-5-mini_Min_v2.csv"),
                os.path.join(base_dir, "GPT-5-mini_Low_v2.csv"),
                os.path.join(base_dir, "GPT-5-mini_Med_v2.csv"),
            ]
            compare_sources = {
                "GPT-4o": [gpt4o_path],
                "GPT-5.2 None": [gpt52_none_path],
                "GPT-5-mini": gpt5mini_v1_paths,
                "GPT-5-mini v2": gpt5mini_v2_paths,
            }
            compare_df = _load_sources(compare_sources)
            if not compare_df.empty:
                compare_grouped = (
                    compare_df.groupby(list(plot_conditions) + ["source"], dropna=False)[list(metrics)]
                    .mean()
                    .reset_index()
                )
                compare_grouped["condition_label"] = compare_grouped[list(plot_conditions)].astype(str).agg("|".join, axis=1)
                compare_order = ["GPT-4o", "GPT-5.2 None", "GPT-5-mini", "GPT-5-mini v2"]
                _plot_two_metric_stack(compare_grouped, compare_order, "Average by Auction Preset")

            low_grouped: pd.DataFrame | None = None
            med_grouped: pd.DataFrame | None = None

            # Graph 2: GPT-5.2 Low vs GPT-5-mini Low vs GPT-5-mini Low v2
            low_sources = {
                "GPT-5.2 Low": [os.path.join(base_dir, "GPT-5.2_Low.csv")],
                "GPT-5-mini Low": [os.path.join(base_dir, "GPT-5-mini_Low.csv")],
                "GPT-5-mini Low v2": [os.path.join(base_dir, "GPT-5-mini_Low_v2.csv")],
            }
            low_df = _load_sources(low_sources)
            if not low_df.empty:
                low_grouped = (
                    low_df.groupby(list(plot_conditions) + ["source"], dropna=False)[list(metrics)]
                    .mean()
                    .reset_index()
                )
                low_grouped["condition_label"] = low_grouped[list(plot_conditions)].astype(str).agg("|".join, axis=1)

            # Graph 3: GPT-5.2 Medium vs GPT-5-mini Med vs GPT-5-mini Med v2
            med_sources = {
                "GPT-5.2 Medium": [os.path.join(base_dir, "GPT-5.2_Medium.csv")],
                "GPT-5-mini Med": [os.path.join(base_dir, "GPT-5-mini_Med.csv")],
                "GPT-5-mini Med v2": [os.path.join(base_dir, "GPT-5-mini_Med_v2.csv")],
            }
            med_df = _load_sources(med_sources)
            if not med_df.empty:
                med_grouped = (
                    med_df.groupby(list(plot_conditions) + ["source"], dropna=False)[list(metrics)]
                    .mean()
                    .reset_index()
                )
                med_grouped["condition_label"] = med_grouped[list(plot_conditions)].astype(str).agg("|".join, axis=1)

            low_order = ["Low: GPT-5.2", "Low: GPT-5-mini", "Low: GPT-5-mini v2"]
            med_order = ["Medium: GPT-5.2", "Medium: GPT-5-mini", "Medium: GPT-5-mini v2"]
            if low_grouped is not None and med_grouped is not None:
                low_combined = low_grouped.copy()
                low_combined["source"] = low_combined["source"].replace(
                    {
                        "GPT-5.2 Low": "Low: GPT-5.2",
                        "GPT-5-mini Low": "Low: GPT-5-mini",
                        "GPT-5-mini Low v2": "Low: GPT-5-mini v2",
                    }
                )
                med_combined = med_grouped.copy()
                med_combined["source"] = med_combined["source"].replace(
                    {
                        "GPT-5.2 Medium": "Medium: GPT-5.2",
                        "GPT-5-mini Med": "Medium: GPT-5-mini",
                        "GPT-5-mini Med v2": "Medium: GPT-5-mini v2",
                    }
                )
                combined_grouped = pd.concat([low_combined, med_combined], ignore_index=True)
                combined_order = low_order + med_order
                _plot_two_metric_stack(
                    combined_grouped,
                    combined_order,
                    "Low + Medium reasoning comparison",
                    regret_ylim=(-0.3, 0.0),
                )
                plt.subplots_adjust(hspace=0.35, top=0.92)
            elif low_grouped is not None:
                _plot_two_metric_stack(
                    low_grouped,
                    ["GPT-5.2 Low", "GPT-5-mini Low", "GPT-5-mini Low v2"],
                    "Low reasoning comparison",
                    regret_ylim=(-0.3, 0.0),
                )
            elif med_grouped is not None:
                _plot_two_metric_stack(
                    med_grouped,
                    ["GPT-5.2 Medium", "GPT-5-mini Med", "GPT-5-mini Med v2"],
                    "Medium reasoning comparison",
                    regret_ylim=(-0.3, 0.0),
                )

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            def _infer_filename(fig) -> str | None:
                titles = []
                for ax in fig.axes:
                    t = ax.get_title()
                    if t:
                        titles.append(t)
                title_blob = " | ".join(titles)

                if "GPT-5-Mini: Avg History for Prompt 1 vs Prompt 2 sorted by Reasoning" in title_blob:
                    return "prompt_compare_truth.png"
                if "GPT-5-Mini: Avg Regret for Prompt 1 vs Prompt 2 sorted by Reasoning" in title_blob:
                    return "prompt_compare_regret.png"
                if ("Mean truth rate" in title_blob) and ("Mean regret" in title_blob):
                    return "history_truncation_ablation.png"
                if ("Average Truth by Auction Preset" in title_blob) and ("Average Regret by Auction Preset" in title_blob):
                    return "avg_by_preset.png"
                if "Average by Auction Preset" in title_blob:
                    return "avg_by_preset_history_only.png"
                return None

            new_figs = [n for n in plt.get_fignums() if n not in existing_figs]
            used = set()
            fallback_idx = 1
            for num in new_figs:
                fig = plt.figure(num)
                name = _infer_filename(fig)
                if not name or name in used:
                    name = f"figure_{fallback_idx}.png"
                    fallback_idx += 1
                used.add(name)
                path = os.path.join(save_dir, f"{save_prefix}{name}")
                fig.savefig(path, dpi=300, bbox_inches="tight")
                if not show:
                    plt.close(fig)

        # Show all generated figures at once (optional).
        if show:
            plt.show()
