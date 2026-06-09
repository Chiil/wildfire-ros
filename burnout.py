"""Fuel burnout (mass-loss) model for coupling Rothermel spread to an LES.

The Rothermel model in :mod:`rothermel` is *quasi-steady*: it returns a rate of
spread and a reaction intensity, but it has no time dimension and no notion of
how long a point keeps burning once the front has passed.  An atmospheric LES,
on the other hand, needs a time-resolved surface **heat flux** as its lower
boundary condition.  WRF-SFIRE supplies this with a separate fuel mass-loss
model layered on top of the spread model; this module reproduces that layer.

Model
-----
For a cell that ignites at time ``t_i`` the remaining dry-fuel fraction decays
exponentially with an e-folding (burnout) time ``T_f``::

    F(t) = exp(-(t - t_i) / T_f)

The sensible heat flux released to the atmosphere is the consumption rate times
the (low) heat of combustion, minus the energy spent vaporising fuel moisture::

    Q(t) = (E_net / T_f) * exp(-(t - t_i) / T_f)        [W/m^2]

where ``E_net`` is the net heat available per unit ground area (J/m^2).

Burnout time scale
------------------
``T_f`` is *not* part of Rothermel.  Two closures are offered:

* ``"anderson"`` (default): the Anderson (1969) flame residence time
  ``tau = 384 / sigma`` minutes, with ``sigma`` the characteristic SAV in
  ft^-1 -- directly available from a :class:`~rothermel.SpreadResult`.  Finer
  fuel burns faster, exactly the SAV dependence Rothermel uses internally.
* ``"wrf"``: WRF-SFIRE's per-fuel ``weight`` parameter, converted with
  ``T_f = weight / 0.85`` seconds.  Pass the weight via ``wrf_weight``.

References
----------
* Anderson, H.E. 1969. Heat transfer and fire spread. USDA For. Serv. Res. Pap.
  INT-69.  (flame residence time tau = 384/sigma)
* Mandel, J.; Beezley, J.D.; Kochanski, A.K. 2011. Coupled atmosphere-wildland
  fire modeling with WRF 3.3 and SFIRE 2011. Geosci. Model Dev. 4, 591-610.
  (the exponential fuel mass-loss / heat-flux coupling used here)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rothermel import HEAT_CONTENT, FuelModel, Moisture

# ---------------------------------------------------------------------------
# Unit conversions and physical constants
# ---------------------------------------------------------------------------
LB_PER_FT2_TO_KG_PER_M2 = 4.882428      # 1 lb/ft^2 -> kg/m^2
BTU_PER_LB_TO_J_PER_KG = 2326.0         # 1 BTU/lb  -> J/kg
LATENT_HEAT_WATER = 2.26e6              # latent heat of vaporisation [J/kg]

# low heat content of all FBFM40 classes, in SI
HEAT_J_PER_KG = HEAT_CONTENT * BTU_PER_LB_TO_J_PER_KG     # 8000 BTU/lb -> J/kg


# ---------------------------------------------------------------------------
# Burnout time-scale closures
# ---------------------------------------------------------------------------
def anderson_residence_time(sav_per_ft: float) -> float:
    """Anderson (1969) flame residence time tau = 384/sigma.

    ``sav_per_ft`` is the characteristic SAV (ft^-1); returns seconds.
    """
    if sav_per_ft <= 0:
        return 0.0
    return 384.0 / sav_per_ft * 60.0          # minutes -> seconds


def wrf_weight_to_tf(weight: float) -> float:
    """WRF-SFIRE fuel ``weight`` parameter -> burnout time T_f = weight/0.85 [s]."""
    return weight / 0.85


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class BurnoutResult:
    """Time-resolved fuel consumption for one cell / fuel model."""

    t_f: float            # e-folding burnout time                 [s]
    load_dry: float       # consumed oven-dry load                 [kg/m^2]
    water_load: float     # fuel moisture mass consumed            [kg/m^2]
    energy_gross: float   # total chemical heat released           [J/m^2]
    energy_latent: float  # energy spent vaporising fuel moisture  [J/m^2]
    energy_net: float     # sensible heat delivered to atmosphere  [J/m^2]
    peak_flux: float      # sensible heat flux at t = t_i          [W/m^2]
    model: str            # 'anderson' | 'wrf' | 'explicit'

    # --- convenience units ---
    @property
    def energy_net_MJ(self) -> float:
        return self.energy_net / 1e6

    @property
    def peak_flux_kW(self) -> float:
        return self.peak_flux / 1e3

    def fuel_fraction(self, t: float) -> float:
        """Remaining dry-fuel fraction F(t) at time t (s) since ignition."""
        if self.t_f <= 0:
            return 0.0
        return math.exp(-t / self.t_f)

    def heat_flux(self, t: float) -> float:
        """Sensible heat flux Q(t) [W/m^2] at time t (s) since ignition."""
        if self.t_f <= 0:
            return 0.0
        return self.energy_net / self.t_f * math.exp(-t / self.t_f)

    def time_series(self, t_end: float | None = None, n: int = 200):
        """Sampled ``(times, fluxes)`` lists from ignition to ``t_end``.

        ``t_end`` defaults to ``5 * T_f`` (~99.3% of the fuel consumed).
        """
        if t_end is None:
            t_end = 5.0 * self.t_f
        if n < 2 or t_end <= 0:
            return [0.0], [self.heat_flux(0.0)]
        dt = t_end / (n - 1)
        times = [i * dt for i in range(n)]
        return times, [self.heat_flux(t) for t in times]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_burnout(
    fm: FuelModel,
    moisture: Moisture,
    characteristic_sav: float | None = None,
    t_f: float | None = None,
    wrf_weight: float | None = None,
    consumed_load: float | None = None,
) -> BurnoutResult:
    """Build the fuel mass-loss heat-release for one fuel model and condition.

    Parameters
    ----------
    fm : FuelModel
    moisture : Moisture                 fuel moisture fractions (for the
                                        evaporative sink)
    characteristic_sav : float          sigma [ft^-1] from a SpreadResult;
                                        required for the default Anderson model
    t_f : float                         explicit burnout time [s]; overrides
                                        every other closure when given
    wrf_weight : float                  WRF-SFIRE ``weight``; used (as
                                        weight/0.85) when ``t_f`` is not given
    consumed_load : float               oven-dry load actually consumed
                                        [lb/ft^2]; defaults to the full fuel-bed
                                        load (all dead + live classes)
    """
    # --- burnout time scale ---
    if t_f is not None:
        tf, model = t_f, "explicit"
    elif wrf_weight is not None:
        tf, model = wrf_weight_to_tf(wrf_weight), "wrf"
    else:
        if characteristic_sav is None:
            raise ValueError(
                "characteristic_sav is required for the Anderson model; pass a "
                "SpreadResult.characteristic_sav, or give t_f / wrf_weight."
            )
        tf, model = anderson_residence_time(characteristic_sav), "anderson"

    # --- consumed dry load (lb/ft^2) ---
    if consumed_load is None:
        consumed_load = fm.w_1h + fm.w_10h + fm.w_100h + fm.w_herb + fm.w_woody

    # --- fuel-moisture water carried by that load (lb/ft^2) ---
    water = (
        fm.w_1h * moisture.m_1h
        + fm.w_10h * moisture.m_10h
        + fm.w_100h * moisture.m_100h
        + fm.w_herb * moisture.m_herb
        + fm.w_woody * moisture.m_woody
    )

    # --- to SI and energy budget ---
    load_dry = consumed_load * LB_PER_FT2_TO_KG_PER_M2          # kg/m^2
    water_load = water * LB_PER_FT2_TO_KG_PER_M2                # kg/m^2
    energy_gross = load_dry * HEAT_J_PER_KG                     # J/m^2
    energy_latent = water_load * LATENT_HEAT_WATER             # J/m^2
    energy_net = max(0.0, energy_gross - energy_latent)
    peak_flux = energy_net / tf if tf > 0 else 0.0

    return BurnoutResult(
        t_f=tf,
        load_dry=load_dry,
        water_load=water_load,
        energy_gross=energy_gross,
        energy_latent=energy_latent,
        energy_net=energy_net,
        peak_flux=peak_flux,
        model=model,
    )
