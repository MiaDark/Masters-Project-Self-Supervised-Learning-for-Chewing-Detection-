"""Download raw data from an S3 bucket into data/raw.

Design goals:
  * Idempotent  -- re-running skips files already present with the same size.
  * Safe        -- never hard-codes credentials; uses boto3's default chain.
  * Inspectable -- supports --dry-run to preview exactly what would be pulled.
  * Filtered    -- optional extension allow-list so you only pull what you need.

Credentials are resolved by boto3 in the usual order: environment variables,
a named AWS_PROFILE, shared ~/.aws/credentials, or an IAM role. Set them with
`aws configure` (recommended) rather than committing keys.

Usage:
    python -m chewing_ssl.data.download                 # use configs/data.yaml
    python -m chewing_ssl.data.download --dry-run        # preview only
    python -m chewing_ssl.data.download --bucket b --prefix p/   # override
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from tqdm import tqdm

from chewing_ssl.utils.config import load_data_config


def _iter_objects(s3_client, bucket: str, prefix: str):
    """Yield every object under a prefix, transparently handling pagination."""
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            # Skip "directory" placeholder keys.
            if not obj["Key"].endswith("/"):
                yield obj


def _wanted(key: str, extensions: list[str]) -> bool:
    """Return True if the key should be downloaded given the extension filter."""
    if not extensions:
        return True
    return any(key.lower().endswith(ext.lower()) for ext in extensions)


def download(
    bucket: str,
    prefix: str,
    dest_root: Path,
    extensions: list[str] | None = None,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Mirror s3://bucket/prefix into dest_root, preserving the key structure.

    Returns (downloaded_count, skipped_count).
    """
    extensions = extensions or []
    s3 = boto3.client("s3")

    # Collect the object list first so we can show an accurate progress bar.
    try:
        objects = [o for o in _iter_objects(s3, bucket, prefix) if _wanted(o["Key"], extensions)]
    except NoCredentialsError:
        sys.exit("No AWS credentials found. Run `aws configure` or set them in .env.")
    except (ClientError, BotoCoreError) as exc:
        sys.exit(f"Failed to list s3://{bucket}/{prefix} -- {exc}")

    if not objects:
        print(f"No matching objects under s3://{bucket}/{prefix}")
        return 0, 0

    downloaded = skipped = 0
    for obj in tqdm(objects, desc="Downloading", unit="file"):
        key = obj["Key"]
        # Strip the prefix so local paths are relative to dest_root.
        rel = key[len(prefix):].lstrip("/") if key.startswith(prefix) else key
        target = dest_root / rel

        if skip_existing and target.exists() and target.stat().st_size == obj["Size"]:
            skipped += 1
            continue

        if dry_run:
            tqdm.write(f"[dry-run] would download {key} -> {target}")
            downloaded += 1
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            s3.download_file(bucket, key, str(target))
            downloaded += 1
        except (ClientError, BotoCoreError) as exc:
            tqdm.write(f"WARNING: failed to download {key} -- {exc}")

    return downloaded, skipped


def main() -> None:
    cfg = load_data_config()
    s3_cfg = cfg.get("s3", {})
    dl_cfg = cfg.get("download", {})

    parser = argparse.ArgumentParser(description="Download raw data from S3.")
    parser.add_argument("--bucket", default=s3_cfg.get("bucket"))
    parser.add_argument("--prefix", default=s3_cfg.get("prefix", ""))
    parser.add_argument("--dest", default=cfg.get("paths", {}).get("raw", "data/raw"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=dl_cfg.get("dry_run", False),
        help="List what would be downloaded without writing files.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download even if a matching file already exists locally.",
    )
    args = parser.parse_args()

    if not args.bucket:
        sys.exit("No bucket given. Set S3_BUCKET in .env, edit configs/data.yaml, or pass --bucket.")

    downloaded, skipped = download(
        bucket=args.bucket,
        prefix=args.prefix,
        dest_root=Path(args.dest),
        extensions=s3_cfg.get("include_extensions", []),
        skip_existing=not args.no_skip_existing and dl_cfg.get("skip_existing", True),
        dry_run=args.dry_run,
    )
    verb = "would download" if args.dry_run else "downloaded"
    print(f"\nDone: {verb} {downloaded} file(s), skipped {skipped} existing.")


if __name__ == "__main__":
    main()
