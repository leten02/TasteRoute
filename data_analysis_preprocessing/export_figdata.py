"""Export the data behind the EDA and preprocessing figures to small CSV files.

Usage:  python data_analysis_preprocessing/export_figdata.py  (run preprocess.py first)
Input:  data_analysis_preprocessing/results/eda_stats.json
Output: data_analysis_preprocessing/results/figdata/*.csv
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = RES / "figdata"
FIG.mkdir(parents=True, exist_ok=True)


def save(name, df):
    df.to_csv(FIG / f"{name}.csv", index=False)
    print(name, len(df))


s = json.load(open(RES / "eda_stats.json", encoding="utf-8"))
save("e3_ccdf", pd.DataFrame(s["ccdf"]))
save("e4_lorenz", pd.DataFrame(s["lorenz"]))
save("e6_grid", pd.DataFrame(s["grid"])[["lon", "lat", "items", "ratings"]])
save("e7_pair_dist", pd.DataFrame(s["pair_dist"]))
save("e8_user_radius", pd.DataFrame(s["user_radius"]))
save("e10_cooc", pd.DataFrame(s["cooc"]))
um = pd.DataFrame(s["user_mean"]).assign(who="users")
im = pd.DataFrame(s["item_mean"]).assign(who="businesses")
save("e11_means", pd.concat([um, im])[["who", "lo", "hi", "n"]])
save("e12_pop_quality", pd.DataFrame(s["pop_quality"]))
save("e13_shrink", pd.DataFrame(s["shrink"]))
save("p2_rating_pre_post", pd.DataFrame(s["rating_pre_post"]))
save("p3_centered", pd.DataFrame(s["centered"]))
save("p5_cats_per_item", pd.DataFrame(s["cats_per_item"]).query("items > 0"))
