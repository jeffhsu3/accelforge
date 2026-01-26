from collections.abc import Iterable, Set, Sequence

import matplotlib.axes as axes
import matplotlib.pyplot as plt
import pandas as pd

from fastfusion.mapper.FFM import Mappings
from fastfusion.mapper.FFM._pareto_df.df_convention import col2energy
from fastfusion.util._base_analysis_types import ActionKey, VerboseActionKey


def plot_mappings_latency(
    mappings: Iterable[Mappings] | Mappings,
):
    raise NotImplementedError()


def plot_mappings_actions(
    mappings: Iterable[Mappings] | Mappings,
    separate_by: Set[str] = None,
    labels=None,
    ax: axes.Axes = None,
):
    raise NotImplementedError()


def plot_energy_breakdown(
    mappings: Iterable[Mappings] | Mappings,
    separate_by: Sequence[str],
    labels: Iterable[str]=None,
):
    """
    Plot energy breakdown.

    Parameters
    ----------
    mappings:
        A mapping to plot or an iterable of mappings to plot. Each mapping will
        be plotted in a new subplot.
    labels:
        Labels to use for each Mapping class in `mappings`.
    separate_by:
        A list that has elements in {"einsum", "tensor", "component"}. The order
        from left to right will determine grouping of the breakdown.
    """
    mappings = [mappings] if isinstance(mappings, Mappings) else list(mappings)
    n_axes = sum(map(len, (m.data for m in mappings)))

    fig, axes = plt.subplots(n_axes, 1, sharey=True)
    if not isinstance(axes, Sequence):
        axes = [axes]

    axes[0].set_ylabel("Energy (pJ)")

    labels = labels+"-" if labels is not None else [""] * len(mappings)
    assert len(labels) == len(mappings)

    if len(separate_by) == 0:
        raise ValueError("Missing categories by which to breakdown energy")

    idx = 0
    for label, df in zip(labels, (m.data for m in mappings)):
        energy_colnames = [c for c in df.columns if "energy" in c and "Total" not in c]
        bar_components = _get_bar_components(energy_colnames, separate_by)

        for j, (_key, row) in enumerate(df.iterrows()):
            ax = axes[idx]
            idx += 1

            ax.set_title(f"{label}mapping{j}")

            bars = []
            heights = []
            for name, constituents in bar_components:
                name = str(name)
                height = 0
                for colname in constituents:
                    col = df[colname]
                    height += col
                bars.append(name)
                heights.append(height.iloc[0])
            ax.bar(bars, height=heights)
            ax.set_xticklabels(bars, rotation=90)
    return fig, axes


def plot_energy_comparison(mappings: Iterable[Mappings] | Mappings, labels=None):
    """
    Plot the result(s) of mapper or model call(s).

    Parameters
    ----------
    mappings:
        A mapping to plot or an iterable of mappings to plot.
    labels:
        Labels to use for each Mapping class in `mappings`.
    """
    fig, ax = plt.subplots()
    ax.set_ylabel("Energy (pJ)")

    mappings = [mappings] if isinstance(mappings, Mappings) else list(mappings)
    labels = labels+"-" if labels is not None else [""] * len(mappings)
    assert len(labels) == len(mappings)

    for label, df in zip(labels, (m.data for m in mappings)):
        bars = [f"{label}mapping{i}" for i in range(len(df))]
        heights = df["Total<SEP>energy"]
        ax.bar(bars, heights)

    return fig, ax


def _get_bar_components(colnames, separate_by):
    if not separate_by:
        yield "", colnames
        return

    split_colnames = []
    for c in colnames:
        key = col2energy(c)
        if not isinstance(key, VerboseActionKey):
            continue
        split_colnames.append([key.einsum, key.level, key.tensor, key.action, c])
    transposed_colnames = zip(*split_colnames)
    df = pd.DataFrame(
        {k: v for k, v in zip(["einsum", "component", "tensor", "action", "colname"], transposed_colnames)}
    )

    for group, subdf in df.groupby(by=separate_by):
        yield group, subdf["colname"]
