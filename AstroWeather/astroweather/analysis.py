import pandas as pd


def is_good(ts, df, max_seeing, min_tau, duration_hrs):
    in_range = (df["Seeing"] < max_seeing) & (df["Tau"] > min_tau)
    delta = df["LocalTime"] - df["LocalTime"][0]
    delta -= delta.shift().fillna(pd.Timedelta(0))
    delta /= pd.Timedelta(hours=1)
    block = (in_range.shift().fillna(0) != in_range).cumsum()
    good = (delta * in_range).groupby(block).sum().max() >= duration_hrs
    return good, ts

def get_probability(df, *args, **kwargs):
    day = pd.Grouper(freq="24H", key="LocalTime", offset="12H")
    month = pd.Grouper(freq="1M")
    nights = pd.Series(*zip(*[is_good(ts, group, *args, **kwargs)
        for ts, group in df.groupby(day) if len(group) > 0]))
    return nights.groupby(month).mean()
