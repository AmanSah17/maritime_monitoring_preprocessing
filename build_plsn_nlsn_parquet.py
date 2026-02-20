"""
Memory-safe AIS ETL for PLSN/NLSN input parquet.

Creates one consolidated parquet with the exact schema expected by
PLSN/NLSN code paths:
    MMSI, BASEDATETIME, LAT, LON, SOG, COG, NAVSTATUS

Usage example:
python build_plsn_nlsn_parquet.py ^
  --input-glob "F:\\PyTorch_GPU\\maritime_monitoring_preprocessing\\processed_data\\AIS_2020_01_*.csv" ^
  --output-parquet "F:\\PyTorch_GPU\\maritime_monitoring_preprocessing\\interpolated_results\\ais_20200105_20200112_plsn_nlsn_ready.parquet" ^
  --chunk-size 300000 ^
  --compression zstd
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numba import cuda, njit, prange
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Column mapping from raw CSV to expected output schema.
# ---------------------------------------------------------------------------
RAW_TO_STD = {
    "MMSI": "MMSI",
    "BaseDateTime": "BASEDATETIME",
    "LAT": "LAT",
    "LON": "LON",
    "SOG": "SOG",
    "COG": "COG",
    "Status": "NAVSTATUS",
}

STD_COLS = ["MMSI", "BASEDATETIME", "LAT", "LON", "SOG", "COG", "NAVSTATUS"]


@dataclass
class ChunkStats:
    rows_in: int = 0
    rows_out: int = 0
    dropped_null_core: int = 0
    dropped_invalid_geo: int = 0
    dropped_invalid_nav: int = 0
    dropped_invalid_sog: int = 0
    dropped_invalid_cog: int = 0


@dataclass
class RunStats:
    total_rows_in: int = 0
    total_rows_out: int = 0
    total_dropped_null_core: int = 0
    total_dropped_invalid_geo: int = 0
    total_dropped_invalid_nav: int = 0
    total_dropped_invalid_sog: int = 0
    total_dropped_invalid_cog: int = 0
    chunks_processed: int = 0

    def add(self, cs: ChunkStats) -> None:
        self.total_rows_in += cs.rows_in
        self.total_rows_out += cs.rows_out
        self.total_dropped_null_core += cs.dropped_null_core
        self.total_dropped_invalid_geo += cs.dropped_invalid_geo
        self.total_dropped_invalid_nav += cs.dropped_invalid_nav
        self.total_dropped_invalid_sog += cs.dropped_invalid_sog
        self.total_dropped_invalid_cog += cs.dropped_invalid_cog
        self.chunks_processed += 1


# ---------------------------------------------------------------------------
# Numba kernels for row validity checks.
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _cpu_valid_mask(lat: np.ndarray, lon: np.ndarray, sog: np.ndarray, cog: np.ndarray, nav: np.ndarray) -> np.ndarray:
    n = lat.shape[0]
    out = np.zeros(n, dtype=np.uint8)
    for i in prange(n):
        valid_geo = (-90.0 <= lat[i] <= 90.0) and (-180.0 <= lon[i] <= 180.0)
        valid_sog = (0.0 <= sog[i] <= 102.4)  # 102.4 = AIS "not available" upper sentinel
        valid_cog = (0.0 <= cog[i] <= 360.0)
        valid_nav = (0.0 <= nav[i] <= 15.0)
        out[i] = 1 if (valid_geo and valid_sog and valid_cog and valid_nav) else 0
    return out


@cuda.jit
def _gpu_valid_mask_kernel(
    lat: np.ndarray, lon: np.ndarray, sog: np.ndarray, cog: np.ndarray, nav: np.ndarray, out: np.ndarray
) -> None:
    i = cuda.grid(1)
    if i >= lat.size:
        return

    valid_geo = (-90.0 <= lat[i] <= 90.0) and (-180.0 <= lon[i] <= 180.0)
    valid_sog = (0.0 <= sog[i] <= 102.4)
    valid_cog = (0.0 <= cog[i] <= 360.0)
    valid_nav = (0.0 <= nav[i] <= 15.0)
    out[i] = 1 if (valid_geo and valid_sog and valid_cog and valid_nav) else 0


def _gpu_valid_mask(lat: np.ndarray, lon: np.ndarray, sog: np.ndarray, cog: np.ndarray, nav: np.ndarray) -> np.ndarray:
    n = lat.size
    if n == 0:
        return np.empty(0, dtype=np.uint8)

    d_lat = cuda.to_device(lat)
    d_lon = cuda.to_device(lon)
    d_sog = cuda.to_device(sog)
    d_cog = cuda.to_device(cog)
    d_nav = cuda.to_device(nav)
    d_out = cuda.device_array(n, dtype=np.uint8)

    threads = 256
    blocks = (n + threads - 1) // threads
    _gpu_valid_mask_kernel[blocks, threads](d_lat, d_lon, d_sog, d_cog, d_nav, d_out)
    cuda.synchronize()
    return d_out.copy_to_host()


def _build_valid_mask(
    df: pd.DataFrame,
    use_cuda: bool,
    logger: logging.Logger,
) -> np.ndarray:
    lat = df["LAT"].to_numpy(dtype=np.float64, copy=False)
    lon = df["LON"].to_numpy(dtype=np.float64, copy=False)
    sog = df["SOG"].to_numpy(dtype=np.float64, copy=False)
    cog = df["COG"].to_numpy(dtype=np.float64, copy=False)
    nav = df["NAVSTATUS"].to_numpy(dtype=np.float64, copy=False)

    if use_cuda:
        if not cuda.is_available():
            raise RuntimeError("CUDA was requested but not available.")
        return _gpu_valid_mask(lat, lon, sog, cog, nav)
    return _cpu_valid_mask(lat, lon, sog, cog, nav)


def _standardize_columns(chunk: pd.DataFrame) -> pd.DataFrame:
    missing = [raw for raw in RAW_TO_STD if raw not in chunk.columns]
    if missing:
        raise KeyError(f"Missing required columns in input chunk: {missing}")
    work = chunk[list(RAW_TO_STD.keys())].rename(columns=RAW_TO_STD)
    return work


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Core numeric coercions.
    df["MMSI"] = pd.to_numeric(df["MMSI"], errors="coerce")
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df["SOG"] = pd.to_numeric(df["SOG"], errors="coerce")
    df["COG"] = pd.to_numeric(df["COG"], errors="coerce")
    df["NAVSTATUS"] = pd.to_numeric(df["NAVSTATUS"], errors="coerce")
    df["BASEDATETIME"] = pd.to_datetime(df["BASEDATETIME"], errors="coerce", utc=False)

    # Drop rows missing mandatory fields used downstream.
    required = ["MMSI", "BASEDATETIME", "LAT", "LON", "SOG", "COG", "NAVSTATUS"]
    df = df.dropna(subset=required).copy()

    # Memory-friendly casts.
    df["MMSI"] = df["MMSI"].astype(np.int64, copy=False)
    df["LAT"] = df["LAT"].astype(np.float32, copy=False)
    df["LON"] = df["LON"].astype(np.float32, copy=False)
    df["SOG"] = df["SOG"].astype(np.float32, copy=False)
    df["COG"] = df["COG"].astype(np.float32, copy=False)
    df["NAVSTATUS"] = df["NAVSTATUS"].astype(np.int16, copy=False)
    return df


def _clean_chunk(chunk: pd.DataFrame, use_cuda: bool, logger: logging.Logger) -> tuple[pd.DataFrame, ChunkStats]:
    stats = ChunkStats(rows_in=len(chunk))

    work = _standardize_columns(chunk)
    before_drop = len(work)
    work = _coerce_types(work)
    stats.dropped_null_core = before_drop - len(work)
    if work.empty:
        return work, stats

    # Vectorized domain filters with Numba.
    mask = _build_valid_mask(work, use_cuda=use_cuda, logger=logger).astype(bool)
    filtered = work.loc[mask].copy()

    # Diagnostics breakdown.
    invalid_geo = ~((work["LAT"].between(-90, 90)) & (work["LON"].between(-180, 180)))
    invalid_sog = ~work["SOG"].between(0.0, 102.4)
    invalid_cog = ~work["COG"].between(0.0, 360.0)
    invalid_nav = ~work["NAVSTATUS"].between(0, 15)
    stats.dropped_invalid_geo = int(invalid_geo.sum())
    stats.dropped_invalid_sog = int(invalid_sog.sum())
    stats.dropped_invalid_cog = int(invalid_cog.sum())
    stats.dropped_invalid_nav = int(invalid_nav.sum())
    stats.rows_out = len(filtered)
    return filtered, stats


def _iter_csv_chunks(paths: Iterable[str], chunk_size: int):
    usecols = list(RAW_TO_STD.keys())
    dtype_map = {
        "MMSI": "Int64",
        "LAT": "float64",
        "LON": "float64",
        "SOG": "float64",
        "COG": "float64",
        "Status": "float64",
        "BaseDateTime": "string",
    }
    for path in paths:
        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            dtype=dtype_map,
            chunksize=chunk_size,
            low_memory=True,
        ):
            yield path, chunk


def build_parquet(
    input_glob: str,
    output_parquet: str,
    chunk_size: int,
    row_group_size: int,
    compression: str,
    use_cuda: bool,
    force_overwrite: bool,
    log_every_chunks: int,
) -> RunStats:
    logger = logging.getLogger("ais_etl")
    paths = sorted(glob.glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No files match input glob: {input_glob}")

    if os.path.exists(output_parquet):
        if not force_overwrite:
            raise FileExistsError(
                f"Output already exists: {output_parquet}. Use --force-overwrite to replace it."
            )
        os.remove(output_parquet)

    os.makedirs(os.path.dirname(output_parquet) or ".", exist_ok=True)
    logger.info("Input files: %d", len(paths))
    for p in paths:
        logger.info(" - %s", p)

    logger.info("CUDA mode: %s", "enabled" if use_cuda else "disabled (CPU numba)")

    arrow_schema = pa.schema(
        [
            ("MMSI", pa.int64()),
            ("BASEDATETIME", pa.timestamp("ns")),
            ("LAT", pa.float32()),
            ("LON", pa.float32()),
            ("SOG", pa.float32()),
            ("COG", pa.float32()),
            ("NAVSTATUS", pa.int16()),
        ]
    )

    stats = RunStats()
    writer: pq.ParquetWriter | None = None
    start = time.time()

    chunk_iter = _iter_csv_chunks(paths=paths, chunk_size=chunk_size)
    pbar = tqdm(unit="rows", desc="ETL rows")

    try:
        for _, raw_chunk in chunk_iter:
            cleaned, cstats = _clean_chunk(raw_chunk, use_cuda=use_cuda, logger=logger)
            stats.add(cstats)
            pbar.update(cstats.rows_in)

            if cleaned.empty:
                continue

            cleaned = cleaned[STD_COLS]
            table = pa.Table.from_pandas(cleaned, schema=arrow_schema, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    output_parquet,
                    schema=arrow_schema,
                    compression=compression,
                    use_dictionary=True,
                )

            writer.write_table(table, row_group_size=row_group_size)

            if stats.chunks_processed % max(1, log_every_chunks) == 0:
                logger.info(
                    "Chunks=%d rows_in=%d rows_out=%d dropped(null=%d geo=%d nav=%d sog=%d cog=%d)",
                    stats.chunks_processed,
                    stats.total_rows_in,
                    stats.total_rows_out,
                    stats.total_dropped_null_core,
                    stats.total_dropped_invalid_geo,
                    stats.total_dropped_invalid_nav,
                    stats.total_dropped_invalid_sog,
                    stats.total_dropped_invalid_cog,
                )
    finally:
        pbar.close()
        if writer is not None:
            writer.close()

    elapsed = time.time() - start
    logger.info("Wrote parquet: %s", output_parquet)
    logger.info("Elapsed: %.2f sec", elapsed)
    logger.info(
        "Final stats: rows_in=%d rows_out=%d dropped(null=%d geo=%d nav=%d sog=%d cog=%d)",
        stats.total_rows_in,
        stats.total_rows_out,
        stats.total_dropped_null_core,
        stats.total_dropped_invalid_geo,
        stats.total_dropped_invalid_nav,
        stats.total_dropped_invalid_sog,
        stats.total_dropped_invalid_cog,
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PLSN/NLSN-ready AIS parquet from raw daily CSV files (memory-safe chunked ETL)."
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default=r"F:\PyTorch_GPU\maritime_monitoring_preprocessing\processed_data\AIS_2020_01_*.csv",
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        default=r"F:\PyTorch_GPU\maritime_monitoring_preprocessing\interpolated_results\ais_20200105_20200112_plsn_nlsn_ready.parquet",
    )
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--row-group-size", type=int, default=250_000)
    parser.add_argument("--compression", type=str, default="zstd", choices=["zstd", "snappy", "gzip", "brotli"])
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--force-overwrite", action="store_true", default=False)
    parser.add_argument("--log-every-chunks", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    build_parquet(
        input_glob=args.input_glob,
        output_parquet=args.output_parquet,
        chunk_size=int(args.chunk_size),
        row_group_size=int(args.row_group_size),
        compression=args.compression,
        use_cuda=bool(args.use_cuda),
        force_overwrite=bool(args.force_overwrite),
        log_every_chunks=int(args.log_every_chunks),
    )


if __name__ == "__main__":
    main()
