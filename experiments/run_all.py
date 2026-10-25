"""Re-run every experiment in order, then redraw the figures.

Usage:  python experiments/run_all.py   (about 6 minutes on a laptop; MF grid is most of it)
Needs data/processed/ from data_analysis_preprocessing/preprocess.py and the raw
business file in data/raw/ (place attributes for cold_start are extracted first).
"""
import importlib
import time

STEPS = ["baseline", "mf", "item_knn", "cbf_hybrid", "compare_models", "ranking", "route", "popularity", "learned_distance", "bpr", "cold_start", "walking", "temporal", "figures"]


def main():
    """Run this step and write its output (see the module docstring)."""
    t0 = time.time()
    # cold_start needs the place attributes extracted from the raw Yelp file
    from tasteroute import item_extras
    from tasteroute.data import load_items

    if not item_extras.OUT.exists():
        print("\n=== item_extras (place attributes and hours) ===", flush=True)
        item_extras.build(load_items()).to_pickle(item_extras.OUT)
    for name in STEPS:
        t = time.time()
        print(f"\n=== {name} ===", flush=True)
        # importlib.import_module(name) imports experiments/<name>.py by name;
        # this works because Python puts the running script's folder on sys.path.
        importlib.import_module(name).main()
        print(f"--- {name} done in {time.time() - t:.0f}s", flush=True)
    print(f"\nall done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
