"""Global heat-budget terms for the ocean (POP2) and the components it
exchanges energy with: sea ice (CICE), land (CLM) and the atmosphere (CAM).

Every flux term is returned as a global-total power time series [W], so
terms from different model grids can be added and compared without any
per-area normalization ambiguity (ocean m^2 vs Earth m^2). Divide by
EARTH_AREA_M2 to express a term in W per m^2 of Earth surface, the same
normalization iEEI uses. Reservoir states (sea-ice/snow heat content) are
returned in [J]; their time tendency is the corresponding storage flux.

Each compute_* function takes an `open_var(var)` callable that returns a
Dataset containing `var` (already time-shifted and year-filtered), so the
same code serves the full run and the smoke test.
"""
import numpy as np
import xarray as xr

from ieei import CAM_EARTH_RADIUS_M

# CESM shared constants (shr_const_mod), used by POP2, CICE and CAM.
LATENT_HEAT_VAPOR = 2.501e6  # J/kg
LATENT_HEAT_FUSION = 3.337e5  # J/kg
RHO_FRESHWATER = 1.0e3  # kg/m^3
RHO_ICE = 0.917e3  # kg/m^3
EARTH_AREA_M2 = 4.0 * np.pi * CAM_EARTH_RADIUS_M**2
# Typical moist-soil/bedrock volumetric heat capacity; CLM's actual value varies
# with soil texture and water content (roughly +/-30%), so soil storage is approximate.
SOIL_HEAT_CAPACITY = 2.2e6  # J m-3 K-1


def pop_global_total(da, ds):
    """POP T-grid field [W/m^2] -> global ocean total [W]."""
    area_m2 = (ds["TAREA"] / 1e4).where(ds["KMT"] > 0, 0.0)
    return (da.fillna(0.0) * area_m2).sum(["nlat", "nlon"])


def cice_global_total(da, ds):
    """CICE grid-cell-mean field [per m^2] -> global total."""
    area_m2 = ds["tarea"].where(ds["tmask"] > 0, 0.0).fillna(0.0)
    return (da.fillna(0.0) * area_m2).sum(["nj", "ni"])


def clm_global_total(da, ds):
    """CLM field [per m^2 of land] -> global land total."""
    land_area_m2 = (ds["area"] * 1e6 * ds["landfrac"]).fillna(0.0)
    return (da.fillna(0.0) * land_area_m2).sum(["lat", "lon"])


def cam_global_total(da, ds):
    """CAM lat/lon field [W/m^2] -> global total [W] (gw-weighted mean x Earth area)."""
    weights = ds["gw"] * xr.ones_like(ds["lon"])
    return da.weighted(weights).mean(["lat", "lon"]) * EARTH_AREA_M2


def _term(da, units, long_name, source, component):
    # Reduce to an in-memory time series now, one source variable at a time. Building every term
    # lazily and writing them in one to_netcdf call keeps ~200 files open in a single dask graph,
    # beyond xarray's file cache (128), and deadlocked on the HDF5 lock while the cache churned.
    da = da.compute()
    da.attrs = dict(units=units, long_name=long_name, source=source, component=component)
    return da


def compute_ocean_terms(open_pop):
    """POP2 heat fluxes into the ocean [W], positive = ocean gains heat.

    SHF is the total coupler surface heat flux; the next eight terms are its
    components (POP2 forcing_coupled: SHF = QSW + LWDN + LWUP + SENH +
    Lv*EVAP + MELTH - Lf*(SNOW + IOFF)), and SHF_RESIDUAL is SHF minus their
    sum, a check on that decomposition. QFLUX, TFW_T and RESID_T are POP's
    internal (non-coupler) heat source terms.
    """
    def total(var):
        ds = open_pop(var)
        return pop_global_total(ds[var], ds)

    def mass_flux_total(var, latent_heat):
        ds = open_pop(var)
        return pop_global_total(latent_heat * ds[var], ds)

    terms = {}
    direct = {
        "SHF": "Total surface heat flux incl. shortwave (coupler)",
        "SHF_QSW": "Net shortwave heat flux",
        "LWDN_F": "Downward longwave heat flux",
        "LWUP_F": "Upward longwave heat flux",
        "SENH_F": "Sensible heat flux",
        "MELTH_F": "Melt heat flux from sea ice",
    }
    for var, long_name in direct.items():
        terms[var] = _term(total(var), "W", long_name, var, "ocean")

    terms["LATENT"] = _term(
        mass_flux_total("EVAP_F", LATENT_HEAT_VAPOR), "W", "Latent heat flux (Lv * evaporation)", "EVAP_F", "ocean"
    )
    terms["SNOW_MELT"] = _term(
        mass_flux_total("SNOW_F", -LATENT_HEAT_FUSION), "W", "Heat to melt snowfall on the ocean (-Lf * snow)", "SNOW_F", "ocean"
    )
    terms["IOFF_MELT"] = _term(
        mass_flux_total("IOFF_F", -LATENT_HEAT_FUSION), "W", "Heat to melt solid ice runoff (-Lf * ice runoff)", "IOFF_F", "ocean"
    )
    component_names = ["SHF_QSW", "LWDN_F", "LWUP_F", "SENH_F", "LATENT", "MELTH_F", "SNOW_MELT", "IOFF_MELT"]
    terms["SHF_RESIDUAL"] = _term(
        terms["SHF"] - sum(terms[n] for n in component_names),
        "W", "SHF minus the sum of its components (decomposition check)", "derived", "ocean",
    )

    qflux_ds = open_pop("QFLUX")
    qflux = qflux_ds["QFLUX"]
    terms["QFLUX"] = _term(
        pop_global_total(qflux, qflux_ds), "W",
        "Internal ocean heat flux due to ice formation (frazil > 0, melt potential < 0)", "QFLUX", "ocean",
    )
    # Monthly means can mix freezing and melting time steps, so this split is approximate.
    terms["QFLUX_POS"] = _term(
        pop_global_total(qflux.clip(min=0.0), qflux_ds), "W",
        "Positive part of monthly-mean QFLUX (frazil heat, approx.)", "QFLUX", "ocean",
    )
    terms["QFLUX_NEG"] = _term(
        pop_global_total(qflux.clip(max=0.0), qflux_ds), "W",
        "Negative part of monthly-mean QFLUX (melt potential, approx.)", "QFLUX", "ocean",
    )
    terms["TFW_T"] = _term(total("TFW_T"), "W", "Heat flux carried by freshwater fluxes", "TFW_T", "ocean")
    terms["RESID_T"] = _term(total("RESID_T"), "W", "Free-surface residual heat flux", "RESID_T", "ocean")
    return terms


def compute_seaice_terms(open_cice):
    """CICE sea-ice energy reservoir [J] and its exchanges with the ocean [W]."""
    terms = {}
    ds = open_cice("sihc")
    terms["ICE_HEAT_CONTENT"] = _term(
        cice_global_total(ds["sihc"], ds), "J", "Sea-ice heat content (relative to 0 C seawater)", "sihc", "seaice"
    )
    ds = open_cice("sisnhc")
    terms["SNOW_ON_ICE_HEAT_CONTENT"] = _term(
        cice_global_total(ds["sisnhc"], ds), "J", "Heat content of snow on sea ice", "sisnhc", "seaice"
    )
    ds = open_cice("fhocn_ai")
    terms["FHOCN_AI"] = _term(
        cice_global_total(ds["fhocn_ai"], ds), "W", "Heat flux from sea ice to ocean (CICE side)", "fhocn_ai", "seaice"
    )
    ds = open_cice("frazil")
    frazil_m_per_s = ds["frazil"] / 100.0 / 86400.0
    terms["FRAZIL_LATENT"] = _term(
        cice_global_total(RHO_ICE * LATENT_HEAT_FUSION * frazil_m_per_s, ds), "W",
        "Latent heat of CICE frazil growth (rho_ice * Lf * frazil rate; approx., ignores ice salinity/temperature)",
        "frazil", "seaice",
    )
    return terms


def compute_land_terms(open_clm):
    """CLM land energy reservoirs [J], the land fluxes that look like (but are
    not) storage [W], and CLM's own energy-conservation errors [W].

    FGR's long-term mean is dominated by the latent heat that melts each
    year's snowfall (FSM), which leaves the land as runoff rather than being
    stored, so land storage comes from the snow and soil *state* variables.
    """
    terms = {}
    for var, long_name in {
        "FGR": "Heat flux into soil/snow incl. snow melt (NOT a storage term)",
        "FSM": "Snow melt heat flux",
        "ERRSEB": "CLM surface energy conservation error",
        "ERRSOI": "CLM soil/lake energy conservation error",
        "ERRSOL": "CLM solar radiation conservation error",
    }.items():
        ds = open_clm(var)
        terms[var] = _term(clm_global_total(ds[var], ds), "W", long_name, var, "land")

    ds = open_clm("H2OSNO")  # mm of liquid water == kg/m^2
    terms["LAND_SNOW_LATENT_CONTENT"] = _term(
        clm_global_total(-LATENT_HEAT_FUSION * ds["H2OSNO"], ds), "J",
        "Latent heat content of the land snow pack (-Lf * snow water equivalent)", "H2OSNO", "land",
    )
    ds = open_clm("TSOI")
    column_K_m = (ds["TSOI"].fillna(0.0) * ds["DZSOI"].fillna(0.0)).sum("levgrnd")
    terms["LAND_SOIL_HEAT_CONTENT"] = _term(
        clm_global_total(SOIL_HEAT_CAPACITY * column_K_m, ds), "J",
        "Soil + bedrock heat content to 42 m (approx.: fixed volumetric heat capacity; use only its trend)",
        "TSOI, DZSOI", "land",
    )
    return terms


def compute_atmosphere_terms(open_cam):
    """CAM top-of-model, top-of-atmosphere and surface energy fluxes [W],
    positive downward (into the system at TOA, into the surface at the
    surface), plus global-mean surface temperature [K]."""
    def total(var):
        ds = open_cam(var)
        return cam_global_total(ds[var], ds)

    terms = {}
    direct = {
        "FSNT": "Net shortwave at top of model",
        "FLNT": "Net longwave at top of model (upward)",
        "FSNTOA": "Net shortwave at top of atmosphere",
        "FLUT": "Upwelling longwave at top of atmosphere",
        "FSNS": "Net shortwave at surface",
        "FLNS": "Net longwave at surface (upward)",
        "LHFLX": "Surface latent heat flux (upward)",
        "SHFLX": "Surface sensible heat flux (upward)",
    }
    for var, long_name in direct.items():
        terms[var] = _term(total(var), "W", long_name, var, "atmosphere")

    terms["TOA_NET_MODEL"] = _term(terms["FSNT"] - terms["FLNT"], "W", "FSNT - FLNT (iEEI definition)", "derived", "atmosphere")
    terms["TOA_NET_TOA"] = _term(terms["FSNTOA"] - terms["FLUT"], "W", "FSNTOA - FLUT", "derived", "atmosphere")

    snowfall = None
    for var in ("PRECSC", "PRECSL"):
        ds = open_cam(var)
        t = cam_global_total(ds[var], ds)
        snowfall = t if snowfall is None else snowfall + t
    terms["SNOWFALL_LATENT"] = _term(
        -LATENT_HEAT_FUSION * RHO_FRESHWATER * snowfall, "W",
        "Latent heat to melt snowfall (-Lf * rho_w * (PRECSC + PRECSL))", "PRECSC, PRECSL", "atmosphere",
    )
    terms["SFC_NET"] = _term(
        terms["FSNS"] - terms["FLNS"] - terms["LHFLX"] - terms["SHFLX"] + terms["SNOWFALL_LATENT"],
        "W", "Net energy flux into the surface (FSNS - FLNS - LHFLX - SHFLX - Lf*snowfall)", "derived", "atmosphere",
    )

    ds = open_cam("TS")
    weights = ds["gw"] * xr.ones_like(ds["lon"])
    terms["TS_GLOBAL_MEAN"] = _term(
        ds["TS"].weighted(weights).mean(["lat", "lon"]), "K", "Global-mean surface temperature", "TS", "atmosphere"
    )
    return terms
