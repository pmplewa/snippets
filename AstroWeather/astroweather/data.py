import pandas as pd

from .utils import extent


def load_dimm_data(path):
    # import DIMM measurements from archive dump
    dimm_data = pd.read_csv(
        filepath_or_buffer=path,
        encoding="utf-8-sig",
        header=None,
        index_col=0,
        parse_dates=True,
        names=[
            "DateTime",
            "Interval",
            "Seeing",
            "Ra",
            "Dec",
            "Airmass",
            "Tau",
            "RelFluxRMS",
            "Theta"])

    # not needed
    dimm_data.drop("Interval", axis=1, inplace=True)
    dimm_data.drop(["Ra", "Dec"], axis=1, inplace=True)

    # convert units
    dimm_data["Seeing"] /= 100
    dimm_data["Airmass"] /= 100
    dimm_data["Tau"] /= 1000
    dimm_data["RelFluxRMS"] /= 100000
    dimm_data["Theta"] /= 100

    return dimm_data

def load_meteo_data(path):
    # import meteorology measurements from archive dump
    meteo_data = pd.read_csv(
        filepath_or_buffer=path,
        encoding="utf-8-sig",
        header=None,
        index_col=0,
        parse_dates=True,
        names=[
            "DateTime",
            "Interval",
            "Temp1",
            "Temp2",
            "Temp3",
            "TempDew1",
            "TempDew2",
            "Rhum1",
            "Rhum2",
            "Pres",
            "PresQNH",
            "WindSpeed1",
            "WindSpeed2",
            "WindDir1",
            "WindDir2",
            "WindSpeedU",
            "WindSpeedV",
            "WindSpeedW",
            "DUS1",
            "DUL1",
            "DUS2",
            "DUL2"])

    meteo_data.drop_duplicates(inplace=True)

    # not needed
    meteo_data.drop("Interval", axis=1, inplace=True)
    meteo_data.drop("PresQNH", axis=1, inplace=True)
    meteo_data.drop(["WindSpeed1", "WindSpeed2"], axis=1, inplace=True)
    meteo_data.drop(["WindDir1", "WindDir2"], axis=1, inplace=True)
    meteo_data.drop(["DUS1", "DUS2"], axis=1, inplace=True)
    meteo_data.drop(["DUL1", "DUL2"], axis=1, inplace=True)

    # convert units
    meteo_data[["Temp1", "Temp2", "Temp3"]] /= 100
    meteo_data[["TempDew1", "TempDew2"]] /= 100
    meteo_data["Pres"] /= 100
    meteo_data[["WindSpeedU", "WindSpeedV", "WindSpeedW"]] /= 100

    return meteo_data

def tz_localize(df):
    dst_opts = dict(nonexistent="NaT", ambiguous="NaT")
    df["LocalTime"] = df.index.tz_localize("America/Santiago", **dst_opts)
    return df

def load_data(path_to_dimm_data, path_to_meteo_data, freq="20T"):
    dimm_data = load_dimm_data(path_to_dimm_data)
    meteo_data = load_meteo_data(path_to_meteo_data)

    # merge DIMM and meteorology data
    data = pd.concat((dimm_data, meteo_data), axis=1)

    def reduce(ts, df):
        tmin, tmax = extent(df.dropna(subset=["Seeing", "Tau"]).index)
        df = df[tmin:tmax]
        df = df.interpolate(method="time", limit=5).dropna()
        df = df.rolling(freq).mean()
        df = df.resample(freq).mean()
        return df.dropna()

    # interpolate measurements to 20-min intervals, per night
    day = pd.Grouper(freq="24H", key="LocalTime", base=12)
    data = tz_localize(pd.concat([reduce(ts, group.drop("LocalTime", axis=1))
      for ts, group in tz_localize(data).groupby(day)]))

    return data
