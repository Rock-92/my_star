from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from astroquery.gaia import Gaia


MAG_LIMIT = 12.0
RA_MIN_DEG = 0.0
RA_MAX_DEG = 360.0
DEC_MIN_DEG = -90.0
DEC_MAX_DEG = 90.0
RA_STEP_DEG = 20.0
DEC_STEP_DEG = 20.0
MAX_ROWS_PER_CHUNK = 1_000_000

OUTPUT_CSV: Path | None = None
CHUNK_DIR: Path | None = None
QUERY_DELAY_SECONDS = 0.2
QUERY_RETRIES = 5
RETRY_DELAY_SECONDS = 5.0
RETRY_BACKOFF = 2.0

OUTPUT_COLUMNS = (
    "source_id",
    "ra",
    "dec",
    "pmra",
    "pmdec",
    "ref_epoch",
    "phot_g_mean_mag",
)


@dataclass(frozen=True)
class SkyChunk:
    index: int
    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float

    @property
    def name(self) -> str:
        return (
            f"ra{self.ra_min:07.3f}_{self.ra_max:07.3f}"
            f"_dec{self.dec_min:+07.3f}_{self.dec_max:+07.3f}"
        ).replace(".", "p").replace("+", "p").replace("-", "m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a Gaia DR3 magnitude-limited catalogue in RA/Dec chunks, "
            "cache each chunk, and merge them into one offline CSV."
        )
    )
    parser.add_argument("--mag-limit", type=float, default=MAG_LIMIT)
    parser.add_argument("--ra-min", type=float, default=RA_MIN_DEG)
    parser.add_argument("--ra-max", type=float, default=RA_MAX_DEG)
    parser.add_argument("--dec-min", type=float, default=DEC_MIN_DEG)
    parser.add_argument("--dec-max", type=float, default=DEC_MAX_DEG)
    parser.add_argument("--ra-step", type=float, default=RA_STEP_DEG)
    parser.add_argument("--dec-step", type=float, default=DEC_STEP_DEG)
    parser.add_argument("--max-rows-per-chunk", type=int, default=MAX_ROWS_PER_CHUNK)
    parser.add_argument("--chunk-dir", type=Path, default=CHUNK_DIR)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--query-delay-seconds", type=float, default=QUERY_DELAY_SECONDS)
    parser.add_argument("--query-retries", type=int, default=QUERY_RETRIES)
    parser.add_argument("--retry-delay-seconds", type=float, default=RETRY_DELAY_SECONDS)
    parser.add_argument("--retry-backoff", type=float, default=RETRY_BACKOFF)
    parser.add_argument("--force", action="store_true", help="Redownload chunks even if cached files already exist.")
    parser.add_argument("--only-merge", action="store_true", help="Skip downloads and only merge cached chunks.")
    parser.add_argument("--no-merge", action="store_true", help="Download chunks but do not merge at the end.")
    args = parser.parse_args()
    apply_derived_paths(args)
    return args


def format_number_for_name(value: float, digits: int = 1) -> str:
    text = f"{float(value):0.{digits}f}"
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def default_dataset_stem(args: argparse.Namespace) -> str:
    return (
        f"gaia_g{format_number_for_name(args.mag_limit)}"
        f"_ra{format_number_for_name(args.ra_min)}_{format_number_for_name(args.ra_max)}"
        f"_dec{format_number_for_name(args.dec_min)}_{format_number_for_name(args.dec_max)}"
        f"_step{format_number_for_name(args.ra_step)}x{format_number_for_name(args.dec_step)}"
        "_pm"
    )


def apply_derived_paths(args: argparse.Namespace) -> None:
    stem = default_dataset_stem(args)
    base_dir = Path(__file__).resolve().parent / "gaia_data"
    if args.output_csv is None:
        args.output_csv = base_dir / f"{stem}.csv"
    if args.chunk_dir is None:
        args.chunk_dir = base_dir / f"{stem}_chunks"
    args.dataset_stem = stem


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.ra_min < args.ra_max <= 360.0:
        raise ValueError("RA range must satisfy 0 <= ra_min < ra_max <= 360")
    if not -90.0 <= args.dec_min < args.dec_max <= 90.0:
        raise ValueError("Dec range must satisfy -90 <= dec_min < dec_max <= 90")
    if args.ra_step <= 0 or args.dec_step <= 0:
        raise ValueError("Chunk steps must be positive")
    if args.max_rows_per_chunk <= 0:
        raise ValueError("--max-rows-per-chunk must be positive")


def build_chunks(args: argparse.Namespace) -> list[SkyChunk]:
    chunks: list[SkyChunk] = []
    index = 0
    dec = float(args.dec_min)
    while dec < args.dec_max - 1e-9:
        next_dec = min(float(args.dec_max), dec + float(args.dec_step))
        ra = float(args.ra_min)
        while ra < args.ra_max - 1e-9:
            next_ra = min(float(args.ra_max), ra + float(args.ra_step))
            chunks.append(SkyChunk(index=index, ra_min=ra, ra_max=next_ra, dec_min=dec, dec_max=next_dec))
            index += 1
            ra = next_ra
        dec = next_dec
    return chunks


def chunk_where(chunk: SkyChunk, mag_limit: float) -> str:
    return (
        f"phot_g_mean_mag <= {float(mag_limit):.6f} "
        f"AND ra >= {chunk.ra_min:.10f} AND ra < {chunk.ra_max:.10f} "
        f"AND dec >= {chunk.dec_min:.10f} AND dec < {chunk.dec_max:.10f}"
    )


def count_query(chunk: SkyChunk, mag_limit: float) -> str:
    return f"""
SELECT COUNT(*) AS n
FROM gaiadr3.gaia_source
WHERE {chunk_where(chunk, mag_limit)}
"""


def data_query(chunk: SkyChunk, mag_limit: float) -> str:
    return f"""
SELECT source_id, ra, dec, pmra, pmdec, ref_epoch, phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE {chunk_where(chunk, mag_limit)}
"""


def launch_table_query(query: str, args: argparse.Namespace):
    attempts = max(1, int(args.query_retries) + 1)
    delay = max(0.0, float(args.retry_delay_seconds))
    backoff = max(1.0, float(args.retry_backoff))

    for attempt in range(1, attempts + 1):
        try:
            job = Gaia.launch_job_async(query, output_format="votable_gzip", verbose=False)
            return job.get_results()
        except Exception as exc:
            if attempt >= attempts:
                raise
            print(
                f"[retry] Gaia query failed on attempt {attempt}/{attempts}: "
                f"{type(exc).__name__}: {exc}. Waiting {delay:.1f}s."
            )
            time.sleep(delay)
            delay *= backoff


def count_rows(chunk: SkyChunk, mag_limit: float, args: argparse.Namespace) -> int:
    table = launch_table_query(count_query(chunk, mag_limit), args)
    return int(table["n"][0])


def chunk_paths(chunk_dir: Path, chunk: SkyChunk) -> tuple[Path, Path]:
    return chunk_dir / f"{chunk.name}.csv", chunk_dir / f"{chunk.name}.json"


def cached_chunk_matches(path: Path, args: argparse.Namespace) -> bool:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    config = meta.get("download_config", {})
    if not isinstance(config, dict):
        return False
    return (
        bool(config.get("includes_proper_motion")) is True
        and tuple(config.get("output_columns", ())) == OUTPUT_COLUMNS
        and float(config.get("mag_limit", -1)) == float(args.mag_limit)
    )


def write_empty_chunk(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_COLUMNS)


def download_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_stem": args.dataset_stem,
        "mag_limit": float(args.mag_limit),
        "ra_min": float(args.ra_min),
        "ra_max": float(args.ra_max),
        "dec_min": float(args.dec_min),
        "dec_max": float(args.dec_max),
        "ra_step": float(args.ra_step),
        "dec_step": float(args.dec_step),
        "max_rows_per_chunk": int(args.max_rows_per_chunk),
        "output_columns": list(OUTPUT_COLUMNS),
        "includes_proper_motion": True,
    }


def write_meta(path: Path, chunk: SkyChunk, row_count: int, status: str, args: argparse.Namespace) -> None:
    payload = asdict(chunk) | {
        "row_count": int(row_count),
        "status": status,
        "download_config": download_config(args),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def download_chunk(chunk: SkyChunk, args: argparse.Namespace) -> int:
    chunk_csv, chunk_meta = chunk_paths(args.chunk_dir, chunk)
    if chunk_csv.exists() and chunk_meta.exists() and not args.force:
        if not cached_chunk_matches(chunk_meta, args):
            raise RuntimeError(
                f"Cached chunk metadata does not match current download format: {chunk_meta}. "
                "Use a new --chunk-dir or pass --force."
            )
        meta = json.loads(chunk_meta.read_text(encoding="utf-8"))
        print(f"[skip] {chunk.index:05d} {chunk.name}: {meta.get('row_count', '?')} rows")
        return int(meta.get("row_count", 0))

    expected_count = count_rows(chunk, args.mag_limit, args)
    if expected_count > args.max_rows_per_chunk:
        raise RuntimeError(
            f"Chunk {chunk.name} contains {expected_count} rows, which is above "
            f"--max-rows-per-chunk={args.max_rows_per_chunk}. Reduce --ra-step/--dec-step."
        )

    print(f"[download] {chunk.index:05d} {chunk.name}: {expected_count} rows")
    if expected_count == 0:
        write_empty_chunk(chunk_csv)
        write_meta(chunk_meta, chunk, 0, "ok", args)
        return 0

    table = launch_table_query(data_query(chunk, args.mag_limit), args)
    df = table.to_pandas()
    actual_count = len(df)
    if actual_count != expected_count:
        raise RuntimeError(
            f"Chunk {chunk.name} expected {expected_count} rows but received {actual_count}. "
            "The query may have been truncated; reduce chunk size and retry."
        )

    tmp_path = chunk_csv.with_suffix(".tmp.csv")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_path, index=False, columns=list(OUTPUT_COLUMNS))
    tmp_path.replace(chunk_csv)
    write_meta(chunk_meta, chunk, actual_count, "ok", args)
    return actual_count


def merge_chunks(chunks: list[SkyChunk], chunk_dir: Path, output_csv: Path) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_csv.with_suffix(".tmp.csv")
    total_rows = 0

    with tmp_path.open("w", encoding="utf-8", newline="") as out_handle:
        writer = csv.writer(out_handle)
        writer.writerow(OUTPUT_COLUMNS)
        for chunk in chunks:
            chunk_csv, chunk_meta = chunk_paths(chunk_dir, chunk)
            if not chunk_csv.exists() or not chunk_meta.exists():
                raise FileNotFoundError(f"Missing cached chunk: {chunk_csv}")
            with chunk_csv.open("r", encoding="utf-8-sig", newline="") as in_handle:
                reader = csv.reader(in_handle)
                header = next(reader, None)
                if header is None:
                    continue
                if tuple(header) != OUTPUT_COLUMNS:
                    raise RuntimeError(
                        f"Cached chunk columns do not match current output format: {chunk_csv}. "
                        f"Expected {OUTPUT_COLUMNS}, got {tuple(header)}."
                    )
                for row in reader:
                    if row:
                        writer.writerow(row)
                        total_rows += 1

    tmp_path.replace(output_csv)
    return total_rows


def write_dataset_metadata(args: argparse.Namespace, chunks: list[SkyChunk], merged_rows: int | None) -> None:
    path = args.output_csv.with_suffix(".json")
    payload = download_config(args) | {
        "chunk_dir": str(args.chunk_dir),
        "output_csv": str(args.output_csv),
        "chunk_count": len(chunks),
        "merged_rows": merged_rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    Gaia.ROW_LIMIT = -1

    chunks = build_chunks(args)
    print(
        f"Gaia DR3 cache download: dataset={args.dataset_stem}, "
        f"G<={args.mag_limit}, RA=[{args.ra_min}, {args.ra_max}), "
        f"Dec=[{args.dec_min}, {args.dec_max}), step={args.ra_step}x{args.dec_step}, "
        f"{len(chunks)} chunks"
    )
    print(
        f"Output columns: {', '.join(OUTPUT_COLUMNS)}"
    )
    print(
        f"Chunk cache: {args.chunk_dir}"
    )
    print(
        f"Merged CSV: {args.output_csv}"
    )
    print(
        "Full all-sky G<=16 can be many GB. Use smaller RA/Dec ranges if you only "
        "need a local offline cache."
    )

    total_rows = 0
    if not args.only_merge:
        args.chunk_dir.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:
            total_rows += download_chunk(chunk, args)
            if args.query_delay_seconds > 0:
                time.sleep(float(args.query_delay_seconds))
        print(f"Cached rows across downloaded/skipped chunks: {total_rows}")

    if not args.no_merge:
        merged_rows = merge_chunks(chunks, args.chunk_dir, args.output_csv)
        print(f"Merged {merged_rows} rows into {args.output_csv}")
        write_dataset_metadata(args, chunks, merged_rows)
    else:
        write_dataset_metadata(args, chunks, None)


if __name__ == "__main__":
    main()
