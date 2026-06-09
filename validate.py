"""Validate the Rothermel implementation against the published FBFM40 table.

The reference JSON carries the authoritative characteristic SAV, packing ratio
and relative packing ratio for every standard fuel model.  Reproducing those
from first principles confirms the fuel-bed weighting is correct.
"""

from rothermel import load_fuel_models, compute_ros, Moisture


def main() -> None:
    models = load_fuel_models()
    # benign moisture/wind just to exercise compute_ros; static properties are
    # condition-independent so any reasonable input reproduces the table values.
    moist = Moisture.from_percent(6, 7, 8, 60, 90)

    print(f"{'code':5} {'sigma':>6} {'ref':>6} {'beta':>8} {'ref':>8} "
          f"{'relPR':>6} {'ref':>6}   status")
    worst = 0.0
    for code, fm in models.items():
        r = compute_ros(fm, moist, wind_midflame_ft_min=0, slope_fraction=0)
        d_sav = abs(r.characteristic_sav - fm.ref_sav)
        d_beta = abs(r.packing_ratio - fm.ref_packing_ratio)
        d_rel = abs(r.relative_packing_ratio - fm.ref_rel_packing_ratio)
        # tolerances reflect rounding in the published table; SAV relative (the
        # published characteristic SAV is itself a rounded, weighted quantity)
        ok = d_sav <= max(2.0, 0.025 * fm.ref_sav) and d_beta <= 5e-5 and d_rel <= 0.01
        worst = max(worst, d_sav)
        print(f"{code:5} {r.characteristic_sav:6.0f} {fm.ref_sav:6.0f} "
              f"{r.packing_ratio:8.5f} {fm.ref_packing_ratio:8.5f} "
              f"{r.relative_packing_ratio:6.2f} {fm.ref_rel_packing_ratio:6.2f}   "
              f"{'OK' if ok else 'MISMATCH'}")
    print(f"\n{len(models)} burnable models checked; "
          f"max characteristic-SAV error = {worst:.2f} ft^-1")


if __name__ == "__main__":
    main()
