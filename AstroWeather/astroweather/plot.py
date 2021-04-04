import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import get_probability
from .utils import minmax_scale


def plot_monthly_seeing(data, **kwargs):
    fig, ax = plt.subplots(**kwargs)

    month = pd.Grouper(freq="1M", key="LocalTime")
    df = data[["LocalTime", "Seeing"]].groupby(month)

    ticks = []
    med_seeing = []
    for i, (ts, group) in enumerate(df):
        if ts.month == 1:
            ticks.append([i, ts.year])
        med_seeing.append(group.median())
        ax.boxplot(
            x=group["Seeing"],
            positions=[i],
            widths=1,
            showfliers=False,
            showbox=False,
            showcaps=False,
            medianprops=dict(alpha=0),
            whiskerprops=dict(linewidth=1, color="gray"))

    ax.plot(med_seeing, color="C00", linewidth=2)

    ax.set_ylim(0, 2.5)
    ax.set_xlabel(None)
    ax.set_ylabel("Seeing [arcsec]")

    ticks, labels = zip(*ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    plt.show()

def plot_seeing_vs_coherence(data, max_seeing=2.5, max_tau=8, **kwargs):
    fig, ax = plt.subplots(**kwargs)

    df = data[(data["Seeing"] < max_seeing) & (data["Tau"] < max_tau)]
    df.plot.hexbin(
        ax=ax,
        x="Seeing",
        y="Tau",
        gridsize=50,
        xlim=(0, 2.5),
        ylim=(0, 8),
        mincnt=1,
        colormap="Blues",
        colorbar=False,
        sharex=False)

    ax.set_xlabel("Seeing [arcsec]")
    ax.set_ylabel("Coherence Time [ms]")

    plt.show()

def plot_seeing_vs_winddir(data, **kwargs):
    wind_dir = np.arctan2(data["WindSpeedV"], data["WindSpeedU"])

    n_bins = 100
    bin_value = np.linspace(-np.pi, np.pi, n_bins)
    bin_index = np.digitize(wind_dir, bin_value)
    bin_count = np.bincount(bin_index)

    med_seeing = data["Seeing"].groupby(bin_index).median()

    fig = plt.figure(**kwargs)
    ax = plt.subplot(projection="polar")

    bars = ax.bar(bin_value, bin_count, width=2*np.pi/n_bins, linewidth=0)

    for c, bar in zip(minmax_scale(med_seeing), bars):
        bar.set_facecolor(plt.cm.Blues(c))

    ax.get_yaxis().set_visible(False)

    plt.show()

def plot_chance_of_seeing(data, **kwargs):
    df = pd.DataFrame({key: get_probability(data, max_seeing=seeing, min_tau=2, duration_hrs=1)
        for key, seeing in [("Fair", 1.2), ("Good", 0.8), ("Excellent", 0.6)]})

    fig, ax = plt.subplots(**kwargs)

    df.plot(
        ax=ax,
        kind="line",
        color="black",
        alpha=0.1,
        legend=None)

    df.rolling(3).mean().plot(
        ax=ax,
        kind="line")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.get_legend().set_bbox_to_anchor((1.02, 1))

    plt.show()
