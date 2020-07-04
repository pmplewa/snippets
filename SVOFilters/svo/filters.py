import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import requests

from astropy.io import votable
#from astropy.modeling.tabular import Tabular1D
from astropy.table import Table
from io import BytesIO
from scipy.interpolate import UnivariateSpline


base_url = "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php"

def query(payload):
    response = requests.get(base_url, params=payload)
    content_stream = BytesIO(response.content)
    table = votable.parse(content_stream)
    status = table.get_info_by_id("QUERY_STATUS")
    assert status.value == "OK", status.value
    return table.get_first_table()

def fetch(filter_id, phot_sys=None):
    payload = {}
    if phot_sys is not None:
        payload.update({"PhotCalID": f"{filter_id}/{phot_sys}"})
    else:
        payload.update({"ID": filter_id})
    table = query(payload)
    return Filter(table)

def parse_range(key, values):
    return {f"{key}_min": values[0], f"{key}_max": values[1]}

def parse_value(key, value):
    return {key: value}

def search(λ_mean=None, λ_eff=None, λ_min=None, λ_max=None, width_eff=None,
           fwhm=None, instrument=None, facility=None, phot_sys=None):
    payload = {}
    if λ_mean is not None:
        payload.update(parse_range("WavelengthMean", λ_mean))
    if λ_eff is not None:
        payload.update(parse_range("WavelengthEff", λ_eff))
    if λ_min is not None:
        payload.update(parse_range("WavelengthMin", λ_min))
    if λ_max is not None:
        payload.update(parse_range("WavelengthMax", λ_max))
    if width_eff is not None:
        payload.update(parse_range("WidthEff", width_eff))
    if fwhm is not None:
        payload.update(parse_range("FWHM", fwhm))
    if instrument is not None:
        payload.update(parse_value("Instrument", instrument))
    if facility is not None:
        payload.update(parse_value("Facility", facility))
    if phot_sys is not None:
        payload.update(parse_value("PhotSystem", phot_sys))
    assert payload
    table = query(payload)
    return table.to_table()

class Filter():
    def __init__(self, table):
        assert isinstance(table, votable.tree.Table)
        try:
            table.get_field_by_id("Wavelength")
            table.get_field_by_id("Transmission")
        except KeyError:
            raise

        # parse parameters
        param_table = []
        for param in table.params:
            param_value = param.value

            if param.datatype == "char":
                param_value = param_value.decode("UTF-8")

            if param.unit is not None:
                param_value *= param.unit

            setattr(self, param.ID, param_value)
            param_table.append([param.ID, param_value])

            if param.description is not None:
                setattr(self, f"{param.ID}_description", param.description)

        assert self.PhotCalID is not None
        assert self.WavelengthMin is not None
        assert self.WavelengthMax is not None
        assert self.WavelengthUnit is not None

        # store parameter and transmission tables
        self._param_table = Table(np.array(param_table), names=("Parameter", "Value"))
        self._transmission_table = table.to_table()

        # interpolation of transmission curve
        #self.model = Tabular1D(points=λ, lookup_table=T)
        self._model = UnivariateSpline(self.λ.data, self.T.data, k=1, s=0, ext=1)

    def __repr__(self):
        return f"<SVO Filter {self.PhotCalID}>"

    @property
    def param_table(self):
        return self._param_table

    @property
    def transmission_table(self):
        return self._transmission_table

    @property
    def model(self):
        return self._model

    @property
    def λ(self):
        return self.transmission_table["Wavelength"]
    
    @property
    def T(self):
        return self.transmission_table["Transmission"]

    def info(self):
        self.param_table.pprint(max_width=-1, max_lines=-1, align=[">", "<"])

    def eval(self, λ):
        if isinstance(λ, u.Quantity):
            λ = λ.to(self.WavelengthUnit).value

        return self.model(λ)

    def integral(self, λ_min, λ_max, normalized=False):
        if isinstance(λ_min, u.Quantity):
            λ_min = λ_min.to(self.WavelengthUnit).value

        if isinstance(λ_max, u.Quantity):
            λ_max = λ_max.to(self.WavelengthUnit).value

        assert λ_max > λ_min

        value = self.model.integral(λ_min, λ_max)

        if normalized:
            λ_min = self.WavelengthMin.to(self.WavelengthUnit).value
            λ_max = self.WavelengthMax.to(self.WavelengthUnit).value
            value /= self.model.integral(λ_min, λ_max)

        return value

    def plot_transmission(self, unit=None, path=None, savefig_kwargs={}):
        fig, ax = plt.subplots()

        ax.set_title(f"{self.filterID}")

        # plot sample points
        λ = self.λ.copy()
        if unit is not None:
            λ = λ.to(unit)
        ax.scatter(λ, self.T)

        # plot model
        λ = np.linspace(self.WavelengthMin, self.WavelengthMax, 500)
        if unit is not None:
            λ = λ.to(unit)
        ax.plot(λ, self.eval(λ))

        ax.set_ylim([0, 1.1])

        ax.set_xlabel(f"Wavelength [{λ.unit.name}]")
        ax.set_ylabel("Transmission")

        if path is not None:
            plt.savefig(path, **savefig_kwargs)

        return fig, ax
