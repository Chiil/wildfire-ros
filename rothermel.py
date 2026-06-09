"""Rothermel (1972) surface fire rate-of-spread for the Scott & Burgan (2005)
standard fire behavior fuel models.

References
----------
* Rothermel, R.C. 1972. A mathematical model for predicting fire spread in
  wildland fuels. USDA For. Serv. Res. Pap. INT-115.
* Albini, F.A. 1976. Estimating wildfire behavior and effects. INT-GTR-30.
* Scott, J.H.; Burgan, R.E. 2005. Standard fire behavior fuel models: a
  comprehensive set for use with Rothermel's surface fire spread model.
  RMRS-GTR-153.  (40 "FBFM40" fuel models; note: often mis-cited as "Burgess".)
* Andrews, P.L. 2018. The Rothermel surface fire spread model and associated
  developments: a comprehensive explanation. RMRS-GTR-371.  (equation listing)

All internal computation is in Rothermel's original imperial units so that the
intermediate parameters (characteristic SAV, packing ratio, reaction intensity,
etc.) match published tables.  Convert the final rate of spread to metric with
the helpers at the bottom.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Physical constants (Rothermel 1972 / Andrews 2018)
# ---------------------------------------------------------------------------
RHO_P = 32.0          # oven-dry fuel particle density        [lb/ft^3]
S_T = 0.0555          # total mineral content                 [fraction]
S_E = 0.01            # effective (silica-free) mineral content[fraction]
HEAT_CONTENT = 8000.0 # low heat content, all FBFM40 classes  [BTU/lb]

# Fixed SAV ratios for the dead time-lag classes                [ft^-1]
SAV_10H = 109.0
SAV_100H = 30.0

# tons/acre -> lb/ft^2
TONS_PER_ACRE_TO_LB_PER_FT2 = 2000.0 / 43560.0   # = 0.0459137

DATA_PATH = Path(__file__).with_name("data") / "fbfm40.json"

# The upstream FBFM40 JSON has a few transcription errors: each listed value
# below disagrees with that same model's own published packing ratio and with
# Scott & Burgan (2005) Table 4.  We correct them here rather than mutate the
# fetched data, so the provenance stays auditable.  Keyed by model code; values
# are JSON paths -> corrected value.
_CORRECTIONS = {
    "SH1": {("fuelLoading", "hundredHour"): 0.0},   # JSON 0.1 t/ac -> 0.0
    "SH6": {("fuelLoading", "hundredHour"): 0.0},   # JSON 0.2 t/ac -> 0.0
    "TL3": {("fuelLoading", "hundredHour"): 2.8},   # JSON 2.9 t/ac -> 2.8
    "SH9": {("fuelBedDepth",): 4.4},                # JSON 3.0 ft   -> 4.4 ft
}

# Class index convention used throughout: 0=1h, 1=10h, 2=100h dead time-lag,
# 3=cured (transferred) herb [dead], 4=live herb, 5=live woody.
DEAD = (0, 1, 2, 3)
LIVE = (4, 5)


# ---------------------------------------------------------------------------
# Fuel model container
# ---------------------------------------------------------------------------
@dataclass
class FuelModel:
    """A single Scott & Burgan (2005) standard fuel model."""

    code: str
    name: str
    description: str
    # oven-dry loads [lb/ft^2]
    w_1h: float
    w_10h: float
    w_100h: float
    w_herb: float
    w_woody: float
    # surface-area-to-volume ratios [ft^-1]
    sav_1h: float
    sav_herb: float
    sav_woody: float
    depth: float          # fuel bed depth                     [ft]
    mx_dead: float        # dead fuel moisture of extinction   [fraction]
    dynamic: bool         # herbaceous load transfers live->dead with curing
    # reference values from the published table (for validation / display)
    ref_sav: float = 0.0
    ref_packing_ratio: float = 0.0
    ref_rel_packing_ratio: float = 0.0

    @property
    def burnable(self) -> bool:
        return self.depth > 0 and (self.w_1h + self.w_herb + self.w_woody) > 0


def load_fuel_models(path: Path | str = DATA_PATH) -> dict[str, FuelModel]:
    """Load the FBFM40 set, returning only the burnable models keyed by code."""
    raw = json.loads(Path(path).read_text())
    models: dict[str, FuelModel] = {}
    c = TONS_PER_ACRE_TO_LB_PER_FT2
    for code, m in raw.items():
        for keypath, value in _CORRECTIONS.get(code, {}).items():
            target = m
            for k in keypath[:-1]:
                target = target[k]
            target[keypath[-1]] = value
        fl = m["fuelLoading"]
        sav = m["sav"]
        fm = FuelModel(
            code=code,
            name=m.get("name", code),
            description=m.get("description", ""),
            w_1h=fl["oneHour"] * c,
            w_10h=fl["tenHour"] * c,
            w_100h=fl["hundredHour"] * c,
            w_herb=fl["liveHerbaceous"] * c,
            w_woody=fl["liveWoody"] * c,
            sav_1h=sav.get("oneHour", 0.0),
            sav_herb=sav.get("liveAndDeadHerbaceous", 0.0),
            sav_woody=sav.get("liveWoody", 0.0),
            depth=m["fuelBedDepth"],
            mx_dead=m["moistureOfExtinction"] / 100.0,
            # dynamic models are exactly those carrying a live herbaceous load
            dynamic=fl["liveHerbaceous"] > 0,
            ref_sav=sav.get("characteristic", 0.0),
            ref_packing_ratio=m.get("packingRatio", 0.0),
            ref_rel_packing_ratio=m.get("relativePackingRatio", 0.0),
        )
        if fm.burnable:
            models[code] = fm
    return models


# ---------------------------------------------------------------------------
# Moisture inputs
# ---------------------------------------------------------------------------
@dataclass
class Moisture:
    """Fuel moisture contents as *fractions* (oven-dry weight basis)."""

    m_1h: float
    m_10h: float
    m_100h: float
    m_herb: float
    m_woody: float

    @classmethod
    def from_percent(cls, m_1h, m_10h, m_100h, m_herb, m_woody) -> "Moisture":
        return cls(m_1h / 100, m_10h / 100, m_100h / 100, m_herb / 100, m_woody / 100)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class SpreadResult:
    ros_ft_min: float          # rate of spread          [ft/min]
    reaction_intensity: float  # I_R                     [BTU/ft^2/min]
    characteristic_sav: float  # sigma_bar               [ft^-1]
    packing_ratio: float       # beta                    [-]
    relative_packing_ratio: float
    propagating_flux_ratio: float
    wind_factor: float         # phi_w
    slope_factor: float        # phi_s
    heat_sink: float           # rho_b * eps * Q_ig      [BTU/ft^3]
    live_mx: float             # live moisture of extinction [fraction]
    cured_fraction: float      # herb curing fraction kappa
    wind_limited: bool         # midflame wind exceeded the reliable limit
    extras: dict = field(default_factory=dict)

    # --- unit conversions for the final answer ---
    @property
    def ros_m_min(self) -> float:
        return self.ros_ft_min * 0.3048

    @property
    def ros_m_s(self) -> float:
        return self.ros_ft_min * 0.3048 / 60.0

    @property
    def ros_km_h(self) -> float:
        return self.ros_m_min * 60.0 / 1000.0


# ---------------------------------------------------------------------------
# The Rothermel model
# ---------------------------------------------------------------------------
def _size_class_bin(sav: float) -> int:
    """Rothermel's 6 surface-area-to-volume size classes (Andrews 2018)."""
    if sav >= 1200:
        return 0
    if sav >= 192:
        return 1
    if sav >= 96:
        return 2
    if sav >= 48:
        return 3
    if sav >= 16:
        return 4
    return 5


def compute_ros(
    fm: FuelModel,
    moisture: Moisture,
    wind_midflame_ft_min: float = 0.0,
    slope_fraction: float = 0.0,
    apply_wind_limit: bool = True,
) -> SpreadResult:
    """Rothermel surface-fire rate of spread for one fuel model and condition.

    Parameters
    ----------
    fm : FuelModel
    moisture : Moisture                 (fractions)
    wind_midflame_ft_min : float        midflame wind speed   [ft/min]
    slope_fraction : float              slope rise/run = tan(slope angle)
    apply_wind_limit : bool             cap phi_w at the Rothermel reliable limit
    """
    # ---- 1. herbaceous curing (dynamic load transfer) --------------------
    if fm.dynamic and fm.w_herb > 0:
        kappa = (1.20 - moisture.m_herb) / 0.90
        kappa = min(1.0, max(0.0, kappa))
    else:
        kappa = 0.0
    w_herb_dead = kappa * fm.w_herb           # transferred to dead, takes 1h moisture
    w_herb_live = (1.0 - kappa) * fm.w_herb

    # per-class arrays  [1h, 10h, 100h, cured-herb, live-herb, live-woody]
    w = [fm.w_1h, fm.w_10h, fm.w_100h, w_herb_dead, w_herb_live, fm.w_woody]
    sav = [fm.sav_1h, SAV_10H, SAV_100H, fm.sav_herb, fm.sav_herb, fm.sav_woody]
    mf = [moisture.m_1h, moisture.m_10h, moisture.m_100h,
          moisture.m_1h, moisture.m_herb, moisture.m_woody]

    # ---- 2. mean fuel-bed properties -------------------------------------
    # surface area weighting (within and between dead/live categories)
    area = [sav[i] * w[i] / RHO_P for i in range(6)]
    a_dead = sum(area[i] for i in DEAD)
    a_live = sum(area[i] for i in LIVE)
    a_tot = a_dead + a_live
    if a_tot <= 0:
        return _zero_result(fm)

    f = [0.0] * 6
    for i in DEAD:
        f[i] = area[i] / a_dead if a_dead > 0 else 0.0
    for i in LIVE:
        f[i] = area[i] / a_live if a_live > 0 else 0.0
    f_dead = a_dead / a_tot
    f_live = a_live / a_tot

    sav_dead = sum(f[i] * sav[i] for i in DEAD)
    sav_live = sum(f[i] * sav[i] for i in LIVE)
    sigma = f_dead * sav_dead + f_live * sav_live          # characteristic SAV

    # bulk density and packing ratio use the full (load-conserving) fuel bed
    w_tot = sum(w)
    rho_b = w_tot / fm.depth
    beta = rho_b / RHO_P
    beta_op = 3.348 * sigma ** -0.8189
    rel_packing = beta / beta_op

    # ---- 3. size-class (g) weighting for net loads -----------------------
    def category_net_load(cat) -> float:
        # g_ij = sum of f over classes sharing the same SAV size-class bin
        bins: dict[int, float] = {}
        for i in cat:
            bins[_size_class_bin(sav[i])] = bins.get(_size_class_bin(sav[i]), 0.0) + f[i]
        return sum(bins[_size_class_bin(sav[i])] * w[i] * (1.0 - S_T) for i in cat)

    wn_dead = category_net_load(DEAD)
    wn_live = category_net_load(LIVE)

    # ---- 4. reaction intensity -------------------------------------------
    a_coef = 133.0 * sigma ** -0.7913
    gamma_max = sigma ** 1.5 / (495.0 + 0.0594 * sigma ** 1.5)
    gamma_prime = gamma_max * rel_packing ** a_coef * math.exp(a_coef * (1.0 - rel_packing))

    # category moisture (area weighted) and damping
    mf_dead = sum(f[i] * mf[i] for i in DEAD)
    mf_live = sum(f[i] * mf[i] for i in LIVE) if a_live > 0 else 0.0

    # live moisture of extinction (Rothermel) using fine-fuel weighting exp(-138/sav)
    num_dead = sum(w[i] * math.exp(-138.0 / sav[i]) for i in DEAD if sav[i] > 0)
    den_live = sum(w[i] * math.exp(-500.0 / sav[i]) for i in LIVE if sav[i] > 0)
    mf_dead_fine = (
        sum(w[i] * math.exp(-138.0 / sav[i]) * mf[i] for i in DEAD if sav[i] > 0) / num_dead
        if num_dead > 0 else 0.0
    )
    w_prime = num_dead / den_live if den_live > 0 else 0.0
    if a_live > 0 and den_live > 0:
        mx_live = 2.9 * w_prime * (1.0 - mf_dead_fine / fm.mx_dead) - 0.226
        mx_live = max(mx_live, fm.mx_dead)
    else:
        mx_live = fm.mx_dead

    eta_m_dead = _moisture_damping(mf_dead, fm.mx_dead)
    eta_m_live = _moisture_damping(mf_live, mx_live) if a_live > 0 else 0.0

    eta_s = 0.174 * S_E ** -0.19      # mineral damping (== 0.4174 for S_e=0.01)
    eta_s = min(1.0, eta_s)

    i_r = gamma_prime * (
        wn_dead * HEAT_CONTENT * eta_m_dead * eta_s
        + wn_live * HEAT_CONTENT * eta_m_live * eta_s
    )

    # ---- 5. propagating flux, wind & slope factors -----------------------
    xi = math.exp((0.792 + 0.681 * math.sqrt(sigma)) * (beta + 0.1)) / (192.0 + 0.2595 * sigma)

    wind_limited = False
    if wind_midflame_ft_min > 0:
        c = 7.47 * math.exp(-0.133 * sigma ** 0.55)
        b = 0.02526 * sigma ** 0.54
        e = 0.715 * math.exp(-3.59e-4 * sigma)
        u = wind_midflame_ft_min
        if apply_wind_limit:
            u_limit = 0.9 * i_r
            if u > u_limit:
                u = u_limit
                wind_limited = True
        phi_w = c * u ** b * rel_packing ** -e
    else:
        phi_w = 0.0

    phi_s = 5.275 * beta ** -0.3 * slope_fraction ** 2 if slope_fraction > 0 else 0.0

    # ---- 6. heat sink and rate of spread ---------------------------------
    def heat_sink_category(cat):
        return sum(f[i] * math.exp(-138.0 / sav[i]) * (250.0 + 1116.0 * mf[i])
                   for i in cat if sav[i] > 0)

    rho_eps_qig = rho_b * (f_dead * heat_sink_category(DEAD)
                           + f_live * heat_sink_category(LIVE))

    if rho_eps_qig <= 0:
        ros = 0.0
    else:
        ros = i_r * xi * (1.0 + phi_w + phi_s) / rho_eps_qig

    return SpreadResult(
        ros_ft_min=ros,
        reaction_intensity=i_r,
        characteristic_sav=sigma,
        packing_ratio=beta,
        relative_packing_ratio=rel_packing,
        propagating_flux_ratio=xi,
        wind_factor=phi_w,
        slope_factor=phi_s,
        heat_sink=rho_eps_qig,
        live_mx=mx_live,
        cured_fraction=kappa,
        wind_limited=wind_limited,
        extras={
            "gamma_prime": gamma_prime,
            "eta_m_dead": eta_m_dead,
            "eta_m_live": eta_m_live,
            "eta_s": eta_s,
            "wn_dead": wn_dead,
            "wn_live": wn_live,
            "beta_op": beta_op,
        },
    )


def _moisture_damping(mf: float, mx: float) -> float:
    if mx <= 0:
        return 0.0
    r = min(mf / mx, 1.0)
    return 1.0 - 2.59 * r + 5.11 * r ** 2 - 3.52 * r ** 3


def _zero_result(fm: FuelModel) -> SpreadResult:
    return SpreadResult(0, 0, 0, 0, 0, 0, 0, 0, 0, fm.mx_dead, 0, False)


# ---------------------------------------------------------------------------
# Wind-speed helpers (input convenience)
# ---------------------------------------------------------------------------
def mps_to_ft_min(v: float) -> float:
    return v * 196.850394          # 1 m/s = 196.85 ft/min


def kmh_to_ft_min(v: float) -> float:
    return v * 1000.0 / 60.0 * 3.280839895


def deg_to_slope_fraction(deg: float) -> float:
    return math.tan(math.radians(deg))


# ---------------------------------------------------------------------------
# Wind adjustment factor (20-ft wind -> midflame wind)
#   Albini & Baughman (1979); Baughman & Albini (1980); Andrews (2012) GTR-266.
#   WAF is the ratio  midflame wind / 20-ft wind.
# ---------------------------------------------------------------------------
# 10-m open wind is ~1.15x the 20-ft open wind (NWCG/BehavePlus convention).
WIND_10M_OVER_20FT = 1.15

# Crown-fill portion threshold separating sheltered from unsheltered (GTR-266).
CROWN_FILL_THRESHOLD = 0.05


def _log_term(h_ft: float) -> float:
    """ln((20 + 0.36 H) / (0.13 H)) -- shared by both WAF models. H in feet."""
    return math.log((20.0 + 0.36 * h_ft) / (0.13 * h_ft))


def waf_unsheltered(fuel_depth_ft: float) -> float:
    """WAF for surface fuel not sheltered by an overstory (GTR-266 eq. 8)."""
    return 1.83 / _log_term(fuel_depth_ft)


def crown_fill_portion(canopy_cover: float, crown_ratio: float) -> float:
    """f = F * CR, with F = CC/3 (GTR-266).  Inputs are fractions (0-1)."""
    return (canopy_cover / 3.0) * crown_ratio


def waf_sheltered(canopy_height_ft: float, canopy_cover: float,
                  crown_ratio: float) -> float:
    """WAF for fuel sheltered beneath a forest canopy (GTR-266 eq. 2).

    canopy_cover and crown_ratio are fractions (0-1); height in feet.
    """
    f = crown_fill_portion(canopy_cover, crown_ratio)
    return 0.555 / (math.sqrt(f * canopy_height_ft) * _log_term(canopy_height_ft))


@dataclass
class WAFResult:
    waf: float
    regime: str          # 'unsheltered' | 'sheltered'
    crown_fill: float     # f


def wind_adjustment_factor(
    fuel_depth_ft: float,
    canopy_cover: float = 0.0,
    canopy_height_ft: float = 0.0,
    crown_ratio: float = 0.0,
) -> WAFResult:
    """Choose and evaluate the WAF model.

    Uses the sheltered model when a canopy is present and the crown-fill
    portion f exceeds 5%; otherwise the unsheltered model.  Canopy cover and
    crown ratio are fractions (0-1).
    """
    f = crown_fill_portion(canopy_cover, crown_ratio)
    if canopy_height_ft > 0 and f > CROWN_FILL_THRESHOLD:
        return WAFResult(waf_sheltered(canopy_height_ft, canopy_cover, crown_ratio),
                         "sheltered", f)
    return WAFResult(waf_unsheltered(fuel_depth_ft), "unsheltered", f)


def wind_10m_to_20ft(v: float) -> float:
    return v / WIND_10M_OVER_20FT


def wind_20ft_to_10m(v: float) -> float:
    return v * WIND_10M_OVER_20FT


if __name__ == "__main__":
    # quick smoke test / validation when run directly
    from validate import main as _validate
    _validate()
