Python 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> """
... Daily Fire-Polygon Weather Collector
... - Loads daily fire polygons (GeoPackage/Shapefile/GeoJSON)
... - Queries Open-Meteo Archive API for daily weather at each polygon centroid
... - Caches progress to allow stop/resume
... - Merges weather back and exports a clean CSV
... 
... Usage (CLI):
...   python weather_collect.py --input /path/to/daily_fire_polygons.gpkg --output-dir ./outputs --sample-size 5000
...   # Optional API key via env var OPEN_METEO_API_KEY (Open-Meteo doesn’t require one by default)
... 
... Requires: geopandas, pandas, numpy, requests, tqdm (optional), pyproj, shapely
... """
... 
... from __future__ import annotations
... import os, json, time, warnings, argparse
... from datetime import datetime
... from pathlib import Path
... from typing import Dict, Any, Optional
... 
... import numpy as np
... import pandas as pd
... import geopandas as gpd
... import requests
... try:
...     from tqdm import tqdm
...     _TQDM = True
... except Exception:
...     _TQDM = False
... 
... warnings.filterwarnings("ignore")
... 
... OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
... WEATHER_VARS = [
...     "temperature_2m_max",
    "temperature_2m_min",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "precipitation_sum",
    "vapour_pressure_deficit_max",
]

RENAME_MAP = {
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "windspeed_10m_max": "wind_max",
    "windgusts_10m_max": "gust_max",
    "precipitation_sum": "precip_sum",
    "vapour_pressure_deficit_max": "vpd_max",
}

def _safe_num(x):
    return pd.to_numeric(x, errors="coerce")

class DailyWeatherCollector:
    def __init__(self, input_path: str, output_dir: str = "./outputs", api_key: Optional[str] = None):
        self.input_path = str(input_path)
        self.output_dir = str(output_dir)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.getenv("OPEN_METEO_API_KEY", None)

        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.weather_results: list[Dict[str, Any]] = []

    # -------------------------
    # Data loading & prep
    # -------------------------
    def load_daily_polygons(self) -> bool:
        print("Loading daily fire polygon data…")
        self.gdf = gpd.read_file(self.input_path)

        # Required columns
        need = ["id", "date", "geometry"]
        miss = [c for c in need if c not in self.gdf.columns]
        if miss:
            print(f"ERROR: Missing required columns: {miss}")
            return False

        # Parse date
        self.gdf["date"] = pd.to_datetime(self.gdf["date"], errors="coerce")

        # WGS84
        if self.gdf.crs is None or str(self.gdf.crs).lower() != "epsg:4326":
            try:
                self.gdf = self.gdf.to_crs(epsg=4326)
            except Exception as e:
                print(f"CRS conversion failed: {e}")
                return False

        # Centroids
        ctr = self.gdf.geometry.centroid
        self.gdf["lat"] = ctr.y
        self.gdf["lon"] = ctr.x

        # Filters
        ok_coords = self.gdf["lat"].between(-90, 90) & self.gdf["lon"].between(-180, 180)
        ok_dates  = self.gdf["date"].between("1940-01-01", "2024-12-31")
        mask = ok_coords & ok_dates
        if (~mask).any():
            print(f"Filtering {(~mask).sum():,} rows with invalid coords/dates")
            self.gdf = self.gdf.loc[mask].copy()

        self.gdf = self.gdf.sort_values(["id", "date"]).reset_index(drop=True)

        print(f"Loaded: {len(self.gdf):,} rows | Fires: {self.gdf['id'].nunique():,}")
        print(f"Date range: {self.gdf['date'].min().date()} → {self.gdf['date'].max().date()}")
        return True

    # -------------------------
    # API call (single day)
    # -------------------------
    def _fetch_weather_one(self, lat: float, lon: float, date_str: str) -> Dict[str, Any]:
        params = {
            "latitude": round(float(lat), 4),
            "longitude": round(float(lon), 4),
            "start_date": date_str,
            "end_date": date_str,
            "daily": ",".join(WEATHER_VARS),
            "timezone": "UTC",
        }
        # Open-Meteo does not require an API key; we keep it for flexibility
        if self.api_key:
            params["apikey"] = self.api_key

        try:
            r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
            if r.status_code != 200:
                return {"success": False, "error": f"HTTP {r.status_code}"}

            data = r.json()
            out = {"success": True}
            if "daily" in data and data["daily"]:
                dd = data["daily"]
                for var in WEATHER_VARS:
                    val = np.nan
                    if var in dd and dd[var]:
                        v0 = dd[var][0]
                        if v0 is not None:
                            val = float(v0)
                    out[var] = val
            return out
        except Exception as e:
            return {"success": False, "error": str(e)}

    # -------------------------
    # Cache helpers
    # -------------------------
    def _cache_path(self) -> str:
        return os.path.join(self.output_dir, "weather_cache.json")

    def _load_cache(self) -> tuple[list[Dict[str, Any]], set[int]]:
        fp = self._cache_path()
        if not os.path.exists(fp):
            return [], set()
        try:
            with open(fp, "r") as f:
                cache = json.load(f)
            results = cache.get("weather_data", [])
            done = {int(x["row_idx"]) for x in results if x.get("success")}
            print(f"Loaded cache with {len(results):,} results ({len(done):,} successful).")
            return results, done
        except Exception:
            print("Cache could not be loaded; starting fresh.")
            return [], set()

    def _save_cache(self, results: list[Dict[str, Any]], failures: list[Dict[str, Any]]):
        payload = {
            "weather_data": results,
            "failures": failures,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with open(self._cache_path(), "w") as f:
            json.dump(payload, f)

    # -------------------------
    # Main collection
    # -------------------------
    def collect(self, sample_size: Optional[int] = None) -> int:
        if self.gdf is None:
            raise RuntimeError("Call load_daily_polygons() first.")

        df = self.gdf
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42).copy().sort_index()

        results, done_idx = self._load_cache()
        failures: list[Dict[str, Any]] = []

        to_iter = df.reset_index().rename(columns={"index": "row_idx"})
        to_iter = to_iter[~to_iter["row_idx"].isin(done_idx)]

        print(f"Collecting weather for {len(to_iter):,} rows (out of {len(df):,}).")

        iterator = to_iter.itertuples(index=False)
        if _TQDM:
            iterator = tqdm(iterator, total=len(to_iter), desc="Weather")

        for k, row in enumerate(iterator, 1):
            row_idx = int(row.row_idx)
            lat, lon, d, fid = float(row.lat), float(row.lon), pd.Timestamp(row.date), str(row.id)
            payload = self._fetch_weather_one(lat, lon, d.strftime("%Y-%m-%d"))
            payload.update({"row_idx": row_idx, "id": fid, "date": d.strftime("%Y-%m-%d"), "lat": lat, "lon": lon})

            if payload.get("success"):
                results.append(payload)
            else:
                failures.append(payload)

            # periodic checkpoint
            if k % 1000 == 0:
                self._save_cache(results, failures)

            # gentle rate limiting for free tier
            if not self.api_key and (k % 100 == 0):
                time.sleep(1.0)

        self._save_cache(results, failures)
        self.weather_results = results
        ok = sum(1 for r in results if r.get("success"))
        print(f"Done. Success: {ok:,} | Failures: {len(failures):,}")
        return ok

    # -------------------------
    # Merge & save
    # -------------------------
    def build_final(self) -> pd.DataFrame:
        if not self.weather_results:
            raise RuntimeError("No weather collected yet.")

        wx = pd.DataFrame([r for r in self.weather_results if r.get("success")]).copy()
        if wx.empty:
            raise RuntimeError("No successful weather rows in cache.")

        # Rename to compact column names
        for old, new in RENAME_MAP.items():
            if old in wx.columns:
                wx[new] = wx[old]

        keep_cols = ["row_idx", "id", "date", "lat", "lon"] + list(RENAME_MAP.values())
        wx = wx[[c for c in keep_cols if c in wx.columns]].copy()

        # Attach back to original rows by row_idx
        gdf = self.gdf.copy()
        gdf["row_idx"] = gdf.reset_index().index  # 0..N-1
        df = pd.merge(gdf.drop(columns=["geometry"], errors="ignore"), wx, on="row_idx", how="left")

        # Derived vars
        if {"temp_max", "temp_min"}.issubset(df.columns):
            df["temp_range"] = df["temp_max"] - df["temp_min"]
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df["month"] = pd.to_datetime(df["date"]).dt.month
        df["fire_season"] = df["month"].isin([6, 7, 8, 9, 10]).astype(int)

        return df

    def save_csv(self, df: pd.DataFrame, filename: str = "daily_fire_weather") -> str:
        out = Path(self.output_dir) / f"{filename}.csv"
        df.to_csv(out, index=False)
        print(f"CSV saved → {out}")
        return str(out)

def main():
    p = argparse.ArgumentParser(description="Collect daily weather for fire polygons via Open-Meteo.")
    p.add_argument("--input", required=True, help="Path to daily fire polygons (gpkg/shp/geojson). Must have columns id, date.")
    p.add_argument("--output-dir", default="./outputs", help="Directory for cache & outputs.")
    p.add_argument("--sample-size", type=int, default=None, help="Optional sample size for quick runs.")
    args = p.parse_args()

    coll = DailyWeatherCollector(input_path=args.input, output_dir=args.output_dir)
    if not coll.load_daily_polygons():
        return

    print("Testing API…")
    test = coll._fetch_weather_one(40.0, -105.0, "2023-07-15")
    if not test.get("success", False):
        print(f"API test failed: {test}")
        return
    print("API test OK.")

    coll.collect(sample_size=args.sample_size)
    final_df = coll.build_final()
    coll.save_csv(final_df)

if __name__ == "__main__":
    main()
