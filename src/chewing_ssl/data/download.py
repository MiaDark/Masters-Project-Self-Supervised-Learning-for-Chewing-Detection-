from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from tqdm import tqdm

from chewing_ssl.utils.config import load_data_config

DEFAULT_PARTICIPANT_PATTERN = r"^p0*(\d+)$"

def parse_int_ranges(spec: str | None) -> set[int] | None:
    """Parse a range spec into a set of ints.

    "1-5,8,10-12" -> {1,2,3,4,5,8,10,11,12};  "all"/""/None -> None (no filter).
    """
    if spec is None:
        return None
    spec = str(spec).strip().lower()
    if spec in ("", "all", "*"):
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def participant_number(folder: str, pattern: str = DEFAULT_PARTICIPANT_PATTERN) -> int | None:
    """Extract the participant integer from a folder name, or None if it doesn't match."""
    m = re.match(pattern, folder)
    return int(m.group(1)) if m else None


def wanted_extension(key: str, extensions: list[str]) -> bool:
    if not extensions:
        return True
    return any(key.lower().endswith(ext.lower()) for ext in extensions)

def list_subfolders(s3, bucket: str, prefix: str) -> list[str]:
    """Return immediate subfolder names (no trailing slash) directly under prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    names: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            names.append(cp["Prefix"][len(prefix):].rstrip("/"))
    return names


def list_objects(s3, bucket: str, prefix: str):
    """Yield every (non-placeholder) object under a prefix, handling pagination."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):
                yield obj


@dataclass
class PlanItem:
    key: str
    size: int
    target: Path


def participant_prefixes(s3, bucket, base_prefix, dataset, ds_cfg) -> list[tuple[str, str]]:
    """Resolve the (label, prefix) of every participant folder to download for a dataset."""

    dataset_prefix = f"{base_prefix}{dataset}/"

    # SussexEatingUk: take every participant folder
    if dataset.lower() == "sussexeatinguk":
        return [
            (folder, f"{dataset_prefix}{folder}/Eating/")
            for folder in list_subfolders(s3, bucket, dataset_prefix)
        ]

    pattern = ds_cfg.get("participant_pattern", DEFAULT_PARTICIPANT_PATTERN)
    wanted = parse_int_ranges(ds_cfg.get("participants"))
    weeks = ds_cfg.get("weeks")

    parents = [f"{dataset_prefix}{w}/" for w in weeks] if weeks else [dataset_prefix]

    selected: list[tuple[str, str]] = []
    for parent in parents:
        for folder in list_subfolders(s3, bucket, parent):
            num = participant_number(folder, pattern)

            if wanted is not None:
                if num is None or num not in wanted:
                    continue

            label = f"{parent[len(dataset_prefix):]}{folder}".strip("/")
            selected.append((label, f"{parent}{folder}/"))

    return selected


def build_plan(s3, cfg, datasets_filter, dest_root) -> dict[str, list[PlanItem]]:
    """Walk the enabled datasets and build a per-dataset list of files to download."""
    s3_cfg = cfg["s3"]
    base_prefix = s3_cfg["base_prefix"]
    extensions = s3_cfg.get("include_extensions", [])
    plan: dict[str, list[PlanItem]] = {}

    for dataset, ds_cfg in cfg.get("datasets", {}).items():
        if not ds_cfg.get("enabled", True):
            continue

        if datasets_filter and dataset not in datasets_filter:
            continue

        parts = participant_prefixes(
            s3,
            s3_cfg["bucket"],
            base_prefix,
            dataset,
            ds_cfg,
        )

        if not parts:
            print(
                f"  WARNING [{dataset}]: 0 participant folders matched "
                f"(participants={ds_cfg.get('participants')!r})"
            )

        items: list[PlanItem] = []

        for _label, prefix in parts:
            for obj in list_objects(s3, s3_cfg["bucket"], prefix):

                if not wanted_extension(obj["Key"], extensions):
                    continue

                rel = obj["Key"][len(base_prefix):]

                items.append(
                    PlanItem(
                        obj["Key"],
                        obj["Size"],
                        dest_root / rel,
                    )
                )

        plan[dataset] = items

    return plan

def _human(num_bytes: int) -> str:
    val = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024 or unit == "TB":
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} TB"


def execute_plan(s3, bucket, plan, skip_existing, dry_run) -> None:
    all_items = [item for items in plan.values() for item in items]
    total_bytes = sum(i.size for i in all_items)

    print("\nDownload plan:")
    for dataset, items in plan.items():
        print(f"  {dataset}: {len(items)} file(s), {_human(sum(i.size for i in items))}")
    print(f"  TOTAL: {len(all_items)} file(s), {_human(total_bytes)}\n")

    if not all_items:
        return

    downloaded = skipped = 0
    for item in tqdm(all_items, desc="Downloading", unit="file", disable=dry_run):
        if skip_existing and item.target.exists() and item.target.stat().st_size == item.size:
            skipped += 1
            continue
        if dry_run:
            print(f"[dry-run] {item.key} -> {item.target}")
            continue
        item.target.parent.mkdir(parents=True, exist_ok=True)
        try:
            s3.download_file(bucket, item.key, str(item.target))
            downloaded += 1
        except (ClientError, BotoCoreError) as exc:
            tqdm.write(f"WARNING: failed {item.key} -- {exc}")

    verb = "would download" if dry_run else "downloaded"
    print(f"\nDone: {verb} {downloaded if dry_run else downloaded} file(s), skipped {skipped}.")


def list_participants(s3, cfg, datasets_filter) -> None:
    """Print the participant folders found per dataset, without downloading."""
    s3_cfg = cfg["s3"]
    base_prefix = s3_cfg["base_prefix"]
    for dataset, ds_cfg in cfg.get("datasets", {}).items():
        if datasets_filter and dataset not in datasets_filter:
            continue
        dataset_prefix = f"{base_prefix}{dataset}/"
        weeks = ds_cfg.get("weeks")
        parents = [f"{dataset_prefix}{w}/" for w in weeks] if weeks else [dataset_prefix]
        print(f"\n{dataset}:")
        for parent in parents:
            label = parent[len(dataset_prefix):].rstrip("/") or "(root)"
            folders = sorted(list_subfolders(s3, s3_cfg["bucket"], parent))
            print(f"  {label}: {len(folders)} folder(s) -> {', '.join(folders) or '(none)'}")


def main() -> None:
    cfg = load_data_config()
    s3_cfg = cfg.get("s3", {})
    dl_cfg = cfg.get("download", {})

    parser = argparse.ArgumentParser(description="Download OCOsense datasets from S3.")
    parser.add_argument("--datasets", nargs="*", help="Subset of dataset names to process.")
    parser.add_argument("--dest", default=cfg.get("paths", {}).get("raw", "data/raw"))
    parser.add_argument("--list", action="store_true", help="List participant folders and exit.")
    parser.add_argument(
        "--dry-run", action="store_true", default=dl_cfg.get("dry_run", False),
        help="List files that would be downloaded without writing them.",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Re-download even if a matching file already exists locally.",
    )
    args = parser.parse_args()

    if not s3_cfg.get("bucket"):
        sys.exit("No bucket configured. Set S3_BUCKET in .env or edit configs/data.yaml.")

    s3 = boto3.client("s3", region_name=s3_cfg.get("region"))

    try:
        if args.list:
            list_participants(s3, cfg, args.datasets)
            return

        plan = build_plan(s3, cfg, args.datasets, Path(args.dest))
        execute_plan(
            s3,
            bucket=s3_cfg["bucket"],
            plan=plan,
            skip_existing=not args.no_skip_existing and dl_cfg.get("skip_existing", True),
            dry_run=args.dry_run,
        )
    except NoCredentialsError:
        sys.exit("No AWS credentials found. Run `aws configure` or set them in .env.")
    except (ClientError, BotoCoreError) as exc:
        sys.exit(f"S3 error: {exc}")


if __name__ == "__main__":
    main()