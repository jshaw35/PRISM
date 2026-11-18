"""
Will Kamp's three-layer energy balance model (EBM), importable from plotting scripts.

The physics below (constants, CO2 record, run_ebm, SAI controllers) is copied unchanged from
cells 3-6 and 11-12 of
    rad_journal_club_paper/PRISM/WorkingModel.ipynb   (commit c286966, 28 Oct 2026)
A script cannot import a notebook, the notebook needs ipywidgets, and the Casper checkout
contains only PRISM/PRISM-jonah-local, so the code and its two CO2 files (data/simple_model/)
live here. If Will changes the model, update this file to match.

Layers (T_a atmosphere, T_u upper ocean 0-200 m, T_l lower ocean >200 m), with t in years:
    C_a dT_a/dt = S(1 - alpha_net(T_a, t)) - eps(CO2(t)) sigma T_a^4 - gamma_au (T_a - T_u)
    C_u dT_u/dt = gamma_au (T_a - T_u) - gamma_ul (T_u - T_l)
    C_l dT_l/dt = gamma_ul (T_u - T_l)
    eps(CO2) = eps0 - 5.35 ln(CO2/CO2_0) / (sigma T0^4)     (Myhre et al. 1998, GRL 25, 2715)

simple_model_annual_fields() is the addition for the J04 figures: it converts a run into annual
means of the quantities J04 plots, using J04's units.
"""

import functools
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from prism_paths import SIMPLE_MODEL_DATA_DIR

# == Simulation window ========================================================
SPINUP_START_YEAR = 1850  # pre-industrial equilibrium year (start of the CO2 record)
RUN_START_YEAR = 1850
RUN_END_YEAR = 2100

# == Physical constants =======================================================
S0 = 1361.0  # Solar constant [W/m2]
S = S0 / 4  # Sphere-averaged mean insolation [W/m2]
SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant [W/m2/K4]

# == Pre-industrial baseline (1850) ===========================================
T0 = 288.15  # Surface temperature [K]  (~15 C)

# == Temperature-dependent earth albedo =======================================
# Budyko-style linear ice-albedo parameterization, alpha_earth(T) = A - B*T, with T the
# atmospheric temperature in kelvin. Clipped to an ice-free floor and a snowball ceiling.
# ALBEDO_A = 1.82  # intercept [-]
# ALBEDO_B = 0.00526  # slope [1/K]

# Trying to match AR6 surface-albedo feedback parameter of 0.35 W/m2/K
# ALBEDO_A = 0.603  # intercept [-]  # new value to match AR6 surface-albedo feedback parameter of 0.35 W/m2/K
# ALBEDO_A = 0.599  # intercept [-]  # tuning for an initial surface albedo of 0.295
# ALBEDO_B = 0.00105 # slope [1/K]  # new value to match AR6 surface-albedo feedback parameter of 0.35 W/m2/K
ALBEDO_A = 1.0167  # intercept [-]  # tuning for an initial surface albedo of 0.295
ALBEDO_B = 0.0025 # slope [1/K]  # new value to match AR6 surface-albedo feedback parameter of 0.35 W/m2/K
ALBEDO_MIN = 0.15  # ice-free lower bound
ALBEDO_MAX = 0.65  # snowball upper bound


# Where is this temperature dependent Earth albedo from? Is it from a paper or a model?
# Ok, it comes trom Budyko's 1969 paper. But does it represent a globally averaged albedo? Or is it a local albedo for the Arctic and Antarctic region?
# Seems like the ice-albedo feedback is too strong under Budyko's parameterization. 
def alpha_earth(T):
    """Temperature-dependent earth albedo [-] (surface + non-SAI atmosphere)."""
    return np.clip(ALBEDO_A - ALBEDO_B * np.asarray(T, dtype=float), ALBEDO_MIN, ALBEDO_MAX)[()]


def alpha_net(T, alpha_sai=0.0):
    """
    Total system albedo with a purely scattering SAI layer above the surface.

    Summing the infinite series of reflections between the surface and the aerosol layer,
        P_abs = (1-alpha_earth)(1-alpha_SAI) / (1 - alpha_earth*alpha_SAI),
    and the system albedo is the reflected complement, alpha_net = 1 - P_abs.
    """
    a_e = alpha_earth(T)
    a_s = np.clip(np.asarray(alpha_sai, dtype=float), 0.0, 0.999)[()]
    return 1.0 - (1.0 - a_e) * (1.0 - a_s) / (1.0 - a_e * a_s)


alpha0 = float(alpha_earth(T0))  # pre-industrial planetary albedo = alpha_earth(T0), no SAI

# == CO2 -> emissivity: Myhre et al. (1998) logarithmic forcing ===============
CO2_0 = 280.0  # pre-industrial CO2 concentration [ppm]
K_CO2 = 5.35  # log-CO2 forcing coefficient [W/m2]

# OLR = eps(CO2) * SIGMA * Ta^4 ; eps0 is set so that CO2=CO2_0 gives T_eq = T0 exactly.
OLR0 = S * (1 - alpha0)  # required pre-industrial OLR [W/m2]
EPS0 = OLR0 / (SIGMA * T0**4)  # pre-industrial effective atmospheric emissivity [-]

# == Layer heat capacities ====================================================
# Ocean layer:  C = rho * c_p * h * f_ocean / (seconds per year)   [W.yr/m2/K]
# Atmosphere :  C = c_p_air * p_s / g       / (seconds per year)
RHO_W = 1025.0  # seawater density [kg/m3]
CP_W = 3990.0  # seawater heat capacity [J/kg/K]
F_OCEAN = 0.71  # ocean fraction of Earth's surface
CP_AIR = 1004.0  # air heat capacity [J/kg/K]
P_SURF = 1.01325e5  # surface pressure [Pa]
G_GRAV = 9.81  # gravity [m/s2]
SEC_PER_YR = 3.1557e7

UPPER_DEPTH = 200.0  # upper-ocean thickness [m]
LOWER_DEPTH = 3500.0  # lower-ocean thickness [m]


def ocean_layer_C(depth_m, f_ocean=F_OCEAN):
    """Heat capacity per m2 of Earth for an ocean layer [W.yr/m2/K]."""
    return RHO_W * CP_W * depth_m * f_ocean / SEC_PER_YR


C_ATMOS = CP_AIR * P_SURF / G_GRAV / SEC_PER_YR  # atmospheric column
C_UPPER = ocean_layer_C(UPPER_DEPTH)  # fast (upper) ocean layer
C_LOWER = ocean_layer_C(LOWER_DEPTH)  # slow (deep) ocean layer

# == Inter-layer heat-exchange coefficients [W/m2/K] ==========================
GAMMA_AU = 5.0  # atmosphere <-> upper ocean
GAMMA_UL = 0.7  # upper <-> lower ocean (deep-ocean heat-uptake efficiency; Geoffroy et al. 2013)

# == CO2 record ===============================================================
#   1850-1978 : Meinshausen et al. (2017) / CMIP6 historical (RCMIP "historical" series)
#   1979-2023 : NOAA GML observed global annual-mean CO2
#   2024-2100 : RCMIP/CMIP6 SSP pathway, offset-anchored to the observed 2023 value
# Sources: https://zenodo.org/records/4589756 and https://gml.noaa.gov/ccgg/trends/gl_data.html
# All SSP scenarios therefore share the same CO2 through 2023.
SSP_SCENARIOS = ["ssp126", "ssp245", "ssp585"]


def _load_co2_data():
    hist = np.genfromtxt(Path(SIMPLE_MODEL_DATA_DIR) / "co2_historical_1850_2023.csv", delimiter=",", skip_header=1)
    ssp = np.genfromtxt(Path(SIMPLE_MODEL_DATA_DIR) / "co2_ssp_scenarios_2024_2100.csv", delimiter=",", skip_header=1)
    hist_years, hist_ppm = hist[:, 0], hist[:, 1]
    ssp_years = ssp[:, 0]
    ssp_cols = {"ssp126": ssp[:, 1], "ssp245": ssp[:, 2], "ssp585": ssp[:, 3]}

    years_by_scenario, ppm_by_scenario = {}, {}
    for name in SSP_SCENARIOS:
        years_by_scenario[name] = np.concatenate([hist_years, ssp_years])
        ppm_by_scenario[name] = np.concatenate([hist_ppm, ssp_cols[name]])
    return years_by_scenario, ppm_by_scenario


_CO2_YEARS, _CO2_PPM = _load_co2_data()

# Derived "hold" scenarios: follow a base SSP pathway until switch_year, then hold CO2 fixed at
# held_ppm from switch_year onward.  name -> (base, switch_year, held_ppm)
_CO2_HOLDS = {}


def register_co2_hold(name, base_scenario, switch_year, held_ppm):
    """Register a scenario that follows base_scenario, then holds CO2 at held_ppm from switch_year."""
    _CO2_HOLDS[name] = (base_scenario, switch_year, float(held_ppm))
    return name


def co2_ppm(t, scenario="ssp245"):
    """
    Atmospheric CO2 concentration [ppm]. 'none' is a pre-industrial control: CO2 held at CO2_0
    for all t. Names from register_co2_hold() follow their base pathway, then hold CO2 fixed.
    """
    if scenario == "none":
        return CO2_0
    if scenario in _CO2_HOLDS:
        base, switch_year, held_ppm = _CO2_HOLDS[scenario]
        return np.where(np.asarray(t) >= switch_year, held_ppm, co2_ppm(t, base))[()]
    return np.interp(t, _CO2_YEARS[scenario], _CO2_PPM[scenario])


# Named hold scenario: SSP2-4.5 until 2035, then CO2 steps back to its 1850 value through 2100.
register_co2_hold("revert1850", "ssp245", 2035, co2_ppm(1850, "ssp245"))


def emissivity(CO2):
    """Effective atmospheric emissivity: decreases as CO2 rises above CO2_0. Vectorized over CO2."""
    return EPS0 - K_CO2 * np.log(CO2 / CO2_0) / (SIGMA * T0**4)


# == Model ====================================================================
def sai_delta_alpha(t, d_alpha=0.0, start_year=2030, ramp_years=10, d_alpha_slope=0.0):
    """
    Albedo of the SAI scattering layer, alpha_SAI(t): 0 before start_year, a linear ramp to d_alpha
    over ramp_years, then d_alpha + d_alpha_slope*(t - ramp_end).
    """
    if d_alpha == 0.0 or t < start_year:
        return 0.0
    ramp_end = start_year + ramp_years
    if t < ramp_end:
        return d_alpha * (t - start_year) / ramp_years
    return d_alpha + d_alpha_slope * (t - ramp_end)


def net_toa_imbalance(t, Ta, d_alpha=0.0, scenario="ssp245", sai_start=2030, sai_ramp=10, d_alpha_slope=0.0):
    """Net top-of-atmosphere flux [W/m2]:  Q = S*(1 - alpha_net(Ta, alpha_SAI(t))) - eps(CO2(t))*sigma*Ta^4."""
    a_sai = sai_delta_alpha(t, d_alpha=d_alpha, start_year=sai_start, ramp_years=sai_ramp, d_alpha_slope=d_alpha_slope)
    alpha = alpha_net(Ta, a_sai)
    eps = emissivity(co2_ppm(t, scenario=scenario))
    return S * (1 - alpha) - eps * SIGMA * Ta**4


def layer_derivs(
    t, y, C_a, C_u, C_l, gamma_au, gamma_ul, d_alpha=0.0, scenario="ssp245", sai_start=2030, sai_ramp=10, d_alpha_slope=0.0
):
    """Right-hand side of the three-layer system.  y = [T_a, T_u, T_l]."""
    Ta, Tu, Tl = y
    R = net_toa_imbalance(t, Ta, d_alpha, scenario, sai_start, sai_ramp, d_alpha_slope)
    f_au = gamma_au * (Ta - Tu)  # atmosphere -> upper ocean
    f_ul = gamma_ul * (Tu - Tl)  # upper ocean -> lower ocean
    return [(R - f_au) / C_a, (f_au - f_ul) / C_u, (f_ul) / C_l]


def _resolve_t_span(t_span):
    """None -> the default simulation window (RUN_START_YEAR, RUN_END_YEAR)."""
    return (RUN_START_YEAR, RUN_END_YEAR) if t_span is None else tuple(t_span)


@functools.lru_cache(maxsize=256)
def initial_state(t_start, C_a=C_ATMOS, C_u=C_UPPER, C_l=C_LOWER, gamma_au=GAMMA_AU, gamma_ul=GAMMA_UL, scenario="ssp245"):
    """
    Model state (T_a, T_u, T_l) at t_start: pre-industrial equilibrium (T0, T0, T0) at or before
    SPINUP_START_YEAR, otherwise the end state of a no-SAI spin-up from SPINUP_START_YEAR.
    """
    if t_start <= SPINUP_START_YEAR:
        return (T0, T0, T0)
    sol = solve_ivp(
        lambda t, y: layer_derivs(t, y, C_a, C_u, C_l, gamma_au, gamma_ul, d_alpha=0.0, scenario=scenario),
        (SPINUP_START_YEAR, t_start),
        [T0, T0, T0],
        method="BDF",
        rtol=1e-8,
        atol=1e-10,
    )
    return tuple(float(v) for v in sol.y[:, -1])


def run_ebm(
    C_a=C_ATMOS,
    C_u=C_UPPER,
    C_l=C_LOWER,
    gamma_au=GAMMA_AU,
    gamma_ul=GAMMA_UL,
    scenario="ssp245",
    d_alpha=0.0,
    sai_start=2030,
    sai_ramp=10,
    d_alpha_slope=0.0,
    t_span=None,
    n_pts=2400,
):
    """
    Integrate the three-layer EBM over t_span, starting from initial_state(t_span[0]). The T_a^4 OLR
    term plus the atmosphere/ocean timescale split makes this stiff, hence the implicit BDF method.

    Returns t [years], Ta [K], Tu [K], Tl [K], Q (net TOA imbalance) [W/m2].
    """
    kw = dict(d_alpha=d_alpha, scenario=scenario, sai_start=sai_start, sai_ramp=sai_ramp, d_alpha_slope=d_alpha_slope)
    args = (C_a, C_u, C_l, gamma_au, gamma_ul)

    t_span = _resolve_t_span(t_span)
    y0 = list(initial_state(t_span[0], *args, scenario=scenario))
    t_eval = np.linspace(*t_span, n_pts)
    sol = solve_ivp(lambda t, y: layer_derivs(t, y, *args, **kw), t_span, y0, t_eval=t_eval, method="BDF", rtol=1e-8, atol=1e-10)

    Ta, Tu, Tl = sol.y
    Q = np.array([net_toa_imbalance(ti, Ta_i, **kw) for ti, Ta_i in zip(sol.t, Ta)])
    return sol.t, Ta, Tu, Tl, Q


# == Closed-loop SAI controllers ==============================================
# ARISE-1.5-style two-phase structure (a simplified analog, not the actual ARISE PI controller):
#   t < CTRL_START                         : no SAI
#   CTRL_START <= t < CTRL_START+CTRL_RAMP : ramp-in, applying a linearly growing fraction of the solved d_alpha
#   t >= CTRL_START+CTRL_RAMP              : full control, re-solving d_alpha every year
CTRL_START = 2035  # year SAI ramp-in begins
CTRL_RAMP = 10  # ramp-in duration [yr]


def alpha_sai_for_absorbed_sw(T, R_target):
    """
    Invert alpha_net(): the SAI-layer albedo that makes the absorbed shortwave equal R_target [W/m2]
    at temperature T. Solving A = (1-a_e)(1-x)/(1 - a_e*x) with A = R_target/S gives
    x = [(1-a_e) - A] / [(1-a_e) - A*a_e]. Clipped to [0, 0.999): SAI can only add reflective aerosol.
    """
    a_e = float(alpha_earth(T))
    A = R_target / S
    denom = (1.0 - a_e) - A * a_e
    if denom <= 0.0:
        return 0.0
    return float(np.clip(((1.0 - a_e) - A) / denom, 0.0, 0.999))


def controller_d_alpha_step(T_target, Tu, t_year, scenario, gamma_au):
    """
    SAI albedo that holds Ta at the fixed T_target, given the current upper-ocean state. The atmosphere
    relaxes well within a year, so setting dTa/dt = 0 at Ta = T_target gives the required absorbed flux,
    R_target = eps(CO2(t))*sigma*T_target^4 + gamma_au*(T_target - Tu).
    """
    eps = emissivity(co2_ppm(t_year, scenario=scenario))
    R_target = eps * SIGMA * T_target**4 + gamma_au * (T_target - Tu)
    return alpha_sai_for_absorbed_sw(T_target, R_target)


def _layer_derivs_fixed_alpha(t, y, C_a, C_u, C_l, gamma_au, gamma_ul, d_alpha, scenario):
    """Three-layer RHS with a single, already-solved albedo offset held fixed."""
    Ta, Tu, Tl = y
    eps = emissivity(co2_ppm(t, scenario=scenario))
    R = S * (1 - alpha_net(Ta, d_alpha)) - eps * SIGMA * Ta**4
    f_au = gamma_au * (Ta - Tu)
    f_ul = gamma_ul * (Tu - Tl)
    return [(R - f_au) / C_a, (f_au - f_ul) / C_u, f_ul / C_l]


def run_controller_ebm(
    C_u=C_UPPER,
    C_l=C_LOWER,
    gamma_au=GAMMA_AU,
    gamma_ul=GAMMA_UL,
    scenario="ssp245",
    t_span=None,
    pts_per_year=8,
    ctrl_start=CTRL_START,
    ctrl_ramp=CTRL_RAMP,
):
    """
    Constant-T_a controller. Integrates one calendar year at a time, re-solving d_alpha at the start of
    each year and holding it fixed for that year. T_target is frozen at T_a when the ramp begins.

    Returns t, Ta, Tu, Tl, Q, d_alpha (all on the same sub-year grid) and T_target [K].
    """
    C_a = C_ATMOS
    t0, t_end = _resolve_t_span(t_span)
    years = np.arange(t0, t_end)

    t_chunks, Ta_chunks, Tu_chunks, Tl_chunks, da_chunks = [], [], [], [], []
    y = list(initial_state(t0, C_a, C_u, C_l, gamma_au, gamma_ul, scenario))
    T_target = None

    for yr in years:
        if yr < ctrl_start:
            d_alpha = 0.0
        else:
            if T_target is None:
                T_target = y[0]  # freeze the target at the state when the ramp begins
            d_alpha_full = controller_d_alpha_step(T_target, y[1], yr, scenario, gamma_au)
            ramp_frac = min(1.0, (yr - ctrl_start + 1) / ctrl_ramp)
            d_alpha = ramp_frac * d_alpha_full

        t_eval = np.linspace(yr, yr + 1, pts_per_year, endpoint=False)
        sol = solve_ivp(
            lambda t, yy: _layer_derivs_fixed_alpha(t, yy, C_a, C_u, C_l, gamma_au, gamma_ul, d_alpha, scenario),
            (yr, yr + 1),
            y,
            t_eval=t_eval,
            method="BDF",
            rtol=1e-8,
            atol=1e-10,
            dense_output=True,
        )

        t_chunks.append(sol.t)
        Ta_chunks.append(sol.y[0])
        Tu_chunks.append(sol.y[1])
        Tl_chunks.append(sol.y[2])
        da_chunks.append(np.full_like(sol.t, d_alpha))
        y = sol.sol(yr + 1)  # true state at the year boundary (t_eval excludes it via endpoint=False)

    t = np.concatenate(t_chunks)
    Ta = np.concatenate(Ta_chunks)
    Tu = np.concatenate(Tu_chunks)
    Tl = np.concatenate(Tl_chunks)
    d_alpha_t = np.concatenate(da_chunks)

    eps_t = emissivity(np.array([co2_ppm(ti, scenario=scenario) for ti in t]))
    Q = S * (1 - alpha_net(Ta, d_alpha_t)) - eps_t * SIGMA * Ta**4
    return t, Ta, Tu, Tl, Q, d_alpha_t, T_target


def controller_d_alpha_step_qbal(Ta, t_year, scenario):
    """SAI albedo that makes S(1-alpha) = OLR (zero TOA imbalance) at the given Ta."""
    eps = emissivity(co2_ppm(t_year, scenario=scenario))
    R_target = eps * SIGMA * Ta**4  # absorbed shortwave must equal OLR
    return alpha_sai_for_absorbed_sw(Ta, R_target)


def run_controller_ebm_qbal(
    C_u=C_UPPER, C_l=C_LOWER, gamma_au=GAMMA_AU, gamma_ul=GAMMA_UL, scenario="ssp245", t_span=None, pts_per_year=1
):
    """
    Zero-imbalance (Q=0) controller. Same yearly structure as run_controller_ebm, but each year solves for
    the d_alpha that zeros Q given T_a at the start of that year, so T_a is free to drift.

    Returns t, Ta, Tu, Tl, Q, d_alpha (all on the same sub-year grid).
    """
    C_a = C_ATMOS
    t0, t_end = _resolve_t_span(t_span)
    years = np.arange(t0, t_end)

    t_chunks, Ta_chunks, Tu_chunks, Tl_chunks, da_chunks = [], [], [], [], []
    y = list(initial_state(t0, C_a, C_u, C_l, gamma_au, gamma_ul, scenario))

    for yr in years:
        if yr < CTRL_START:
            d_alpha = 0.0
        else:
            d_alpha_full = controller_d_alpha_step_qbal(y[0], yr, scenario)
            ramp_frac = min(1.0, (yr - CTRL_START + 1) / CTRL_RAMP)
            d_alpha = ramp_frac * d_alpha_full

        t_eval = np.linspace(yr, yr + 1, pts_per_year, endpoint=False)
        sol = solve_ivp(
            lambda t, yy: _layer_derivs_fixed_alpha(t, yy, C_a, C_u, C_l, gamma_au, gamma_ul, d_alpha, scenario),
            (yr, yr + 1),
            y,
            t_eval=t_eval,
            method="BDF",
            rtol=1e-8,
            atol=1e-10,
            dense_output=True,
        )

        t_chunks.append(sol.t)
        Ta_chunks.append(sol.y[0])
        Tu_chunks.append(sol.y[1])
        Tl_chunks.append(sol.y[2])
        da_chunks.append(np.full_like(sol.t, d_alpha))
        y = sol.sol(yr + 1)  # true state at the year boundary

    t = np.concatenate(t_chunks)
    Ta = np.concatenate(Ta_chunks)
    Tu = np.concatenate(Tu_chunks)
    Tl = np.concatenate(Tl_chunks)
    d_alpha_t = np.concatenate(da_chunks)

    eps_t = emissivity(np.array([co2_ppm(ti, scenario=scenario) for ti in t]))
    Q = S * (1 - alpha_net(Ta, d_alpha_t)) - eps_t * SIGMA * Ta**4
    return t, Ta, Tu, Tl, Q, d_alpha_t


# == Conversion to the quantities J04 plots ===================================
# Allowed values of sai_controller in simple_model_annual_fields (None = no SAI)
SAI_CONTROLLERS = ("constant_Ta", "zero_imbalance")


def simple_model_annual_fields(
    scenario: str = "ssp245", start_year: int = RUN_START_YEAR, end_year: int = RUN_END_YEAR, sai_controller: str = None
) -> dict:
    """
    Run the model and return annual means of the quantities in the J04 figures.

    sai_controller selects the SAI treatment: None (no SAI), "constant_Ta" (run_controller_ebm), or
    "zero_imbalance" (run_controller_ebm_qbal). Controller runs have no SAI before CTRL_START.

    The run is continuous from start_year, so iEEI and OHC are both anomalies relative to start_year
    (zero at the start of 1850 by default). No offset is needed between historical and future periods.

    Returns a dict of 1-D arrays, one value per calendar year, year = start_year .. end_year - 1:
        year         calendar year; the value is the mean over [year, year + 1), as in J04's CESM2 annual means
        asr_W_m2     absorbed shortwave, S(1 - alpha_net(T_a, alpha_SAI))
        olr_W_m2     outgoing longwave, eps(CO2) sigma T_a^4
        eei_W_m2     ASR - OLR (the model's Q)
        ieei_J_m2    time integral of EEI, per m2 of Earth
        ohc_J_m2     upper + lower ocean heat content anomaly, per m2 of Earth (C_u, C_l already
                     include F_OCEAN), the same basis as J04's CESM2 OHC
        ts_K         model atmospheric temperature T_a, the model's only near-surface temperature
        alpha_sai    SAI-layer albedo [-], zero without SAI
        alpha_earth  temperature-dependent earth albedo alpha_earth(T_a) [-] (surface + non-SAI
                     atmosphere; Budyko-style linear fit, see alpha_earth()). Excludes the SAI layer;
                     the total system albedo with SAI is alpha_net(T_a, alpha_sai).
    """
    # Monthly output, matching J04's monthly CESM2 cumulative sum
    n_months = 12 * (end_year - start_year)
    t_span = (start_year, end_year)
    if sai_controller is None:
        # n_months + 1 points so the grid has 12 intervals per year; the end point is dropped below
        t_yr, ta_K, tu_K, tl_K, eei_W_m2 = run_ebm(scenario=scenario, t_span=t_span, n_pts=n_months + 1)
        alpha_sai = np.zeros_like(t_yr)
    elif sai_controller == "constant_Ta":
        t_yr, ta_K, tu_K, tl_K, eei_W_m2, alpha_sai, _ = run_controller_ebm(scenario=scenario, t_span=t_span, pts_per_year=12)
    elif sai_controller == "zero_imbalance":
        t_yr, ta_K, tu_K, tl_K, eei_W_m2, alpha_sai = run_controller_ebm_qbal(scenario=scenario, t_span=t_span, pts_per_year=12)
    else:
        raise ValueError(f"sai_controller={sai_controller!r} must be None or one of {SAI_CONTROLLERS}")

    asr_W_m2 = S * (1.0 - alpha_net(ta_K, alpha_sai))
    olr_W_m2 = asr_W_m2 - eei_W_m2

    # Cumulative trapezoid of EEI [W/m2] over time [yr], converted to J/m2. Zero at t = start_year.
    dt_yr = np.diff(t_yr)
    ieei_J_m2 = np.concatenate([[0.0], np.cumsum(0.5 * (eei_W_m2[1:] + eei_W_m2[:-1]) * dt_yr)]) * SEC_PER_YR

    # Heat content change of the two ocean layers: C [W.yr/m2/K] * dT [K] * [s/yr] = J/m2
    ohc_J_m2 = (C_UPPER * (tu_K - tu_K[0]) + C_LOWER * (tl_K - tl_K[0])) * SEC_PER_YR

    # Annual means over the first n_months points. In a no-SAI run the final point (t = end_year) belongs to the next year.
    year = np.arange(start_year, end_year)

    def annual_mean(monthly_values):
        return monthly_values[:n_months].reshape(-1, 12).mean(axis=1)

    return {
        "year": year,
        "asr_W_m2": annual_mean(asr_W_m2),
        "olr_W_m2": annual_mean(olr_W_m2),
        "eei_W_m2": annual_mean(eei_W_m2),
        "ieei_J_m2": annual_mean(ieei_J_m2),
        "ohc_J_m2": annual_mean(ohc_J_m2),
        "ts_K": annual_mean(ta_K),
        "alpha_sai": annual_mean(alpha_sai),
        # Evaluated on the monthly T_a, then annually averaged like every other field
        "alpha_earth": annual_mean(alpha_earth(ta_K)),
    }
