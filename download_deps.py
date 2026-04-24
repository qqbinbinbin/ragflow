#!/usr/bin/env python3

# PEP 723 metadata
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "nltk",
#   "huggingface-hub"
# ]
# ///

import argparse
import hashlib
import os
import subprocess
import time
import urllib.request
from typing import Union

# huggingface_hub reads this at import time. Keep it before importing the
# package so Aliyun/ECS builds do not prefer the hf-xet client path.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from huggingface_hub import hf_hub_download


ALIYUN_UV_MIRROR = "https://mirrors.aliyun.com/github-release/astral-sh/uv/0.9.16/"
GITHUB_UV_RELEASE = "https://github.com/astral-sh/uv/releases/download/0.9.16/"


def get_urls(use_china_mirrors=False) -> list[Union[str, list[str]]]:
    if use_china_mirrors:
        return [
            "https://mirrors.aliyun.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb",
            "https://mirrors.aliyun.com/ubuntu-ports/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_arm64.deb",
            "https://repo.huaweicloud.com/repository/maven/org/apache/tika/tika-server-standard/3.3.0/tika-server-standard-3.3.0.jar",
            "https://repo.huaweicloud.com/repository/maven/org/apache/tika/tika-server-standard/3.3.0/tika-server-standard-3.3.0.jar.md5",
            "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
            ["https://registry.npmmirror.com/-/binary/chrome-for-testing/121.0.6167.85/linux64/chrome-linux64.zip", "chrome-linux64.zip"],
            ["https://registry.npmmirror.com/-/binary/chrome-for-testing/121.0.6167.85/linux64/chromedriver-linux64.zip", "chromedriver-linux64.zip"],
            [
                f"{ALIYUN_UV_MIRROR}uv-x86_64-unknown-linux-gnu.tar.gz",
                "uv-x86_64-unknown-linux-gnu.tar.gz",
                f"{GITHUB_UV_RELEASE}uv-x86_64-unknown-linux-gnu.tar.gz",
            ],
            [
                f"{ALIYUN_UV_MIRROR}uv-aarch64-unknown-linux-gnu.tar.gz",
                "uv-aarch64-unknown-linux-gnu.tar.gz",
                f"{GITHUB_UV_RELEASE}uv-aarch64-unknown-linux-gnu.tar.gz",
            ],
        ]
    else:
        return [
            "http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb",
            "http://ports.ubuntu.com/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_arm64.deb",
            "https://repo1.maven.org/maven2/org/apache/tika/tika-server-standard/3.3.0/tika-server-standard-3.3.0.jar",
            "https://repo1.maven.org/maven2/org/apache/tika/tika-server-standard/3.3.0/tika-server-standard-3.3.0.jar.md5",
            "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
            ["https://storage.googleapis.com/chrome-for-testing-public/121.0.6167.85/linux64/chrome-linux64.zip", "chrome-linux64.zip"],
            ["https://storage.googleapis.com/chrome-for-testing-public/121.0.6167.85/linux64/chromedriver-linux64.zip", "chromedriver-linux64.zip"],
            "https://github.com/astral-sh/uv/releases/download/0.9.16/uv-x86_64-unknown-linux-gnu.tar.gz",
            "https://github.com/astral-sh/uv/releases/download/0.9.16/uv-aarch64-unknown-linux-gnu.tar.gz",
        ]


repo_files = {
    "InfiniFlow/text_concat_xgb_v1.0": {
        "updown_concat_xgb.model": None,
    },
    "InfiniFlow/deepdoc": {
        "det.onnx": None,
        "layout.onnx": "de401c03ee30b1c120416dc06f0705237f0c36d3cdb692c9bfefe8a8f98a4b70",
        "layout.laws.onnx": None,
        "layout.manual.onnx": None,
        "layout.paper.onnx": None,
        "ocr.res": None,
        "rec.onnx": None,
        "tsr.onnx": None,
    },
}


def parse_csv_env(name):
    raw = os.environ.get(name, "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def filtered_repo_files():
    only_repos = parse_csv_env("RAGFLOW_HF_ONLY_REPOS")
    only_files = parse_csv_env("RAGFLOW_HF_ONLY_FILES")
    selected = {}
    for repo_id, files in repo_files.items():
        if only_repos and repo_id not in only_repos:
            continue
        filtered_files = {
            filename: expected_sha256
            for filename, expected_sha256 in files.items()
            if not only_files or filename in only_files
        }
        if filtered_files:
            selected[repo_id] = filtered_files
    return selected


def compute_sha256(path):
    sha256 = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_file(path, expected_sha256):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    if not expected_sha256:
        return True
    actual_sha256 = compute_sha256(path)
    if actual_sha256 == expected_sha256:
        return True
    print(
        f"SHA256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}",
        flush=True,
    )
    return False


def build_resolve_url(endpoint, repository_id, revision, filename):
    base = endpoint.rstrip("/") if endpoint else "https://huggingface.co"
    return f"{base}/{repository_id}/resolve/{revision}/{filename}"


def download_via_curl(repository_id, revision, filename, target_path, endpoint):
    url = build_resolve_url(endpoint, repository_id, revision, filename)
    print(f"Falling back to curl for {repository_id}/{filename} via {url}", flush=True)
    subprocess.run(
        [
            "curl",
            "-fL",
            "-C",
            "-",
            "--retry",
            os.environ.get("RAGFLOW_HF_CURL_RETRIES", "20"),
            "--retry-delay",
            os.environ.get("RAGFLOW_HF_CURL_RETRY_DELAY", "5"),
            "--retry-all-errors",
            "--connect-timeout",
            os.environ.get("RAGFLOW_HF_CURL_CONNECT_TIMEOUT", "30"),
            "--max-time",
            "0",
            "-o",
            target_path,
            url,
        ],
        check=True,
    )


def download_hf_file(repository_id, revision, filename, expected_sha256, local_directory, endpoint, retries):
    target_path = os.path.join(local_directory, filename)
    if verify_file(target_path, expected_sha256):
        print(f"Using existing huggingface file {repository_id}/{filename}", flush=True)
        return

    if os.path.exists(target_path):
        os.remove(target_path)

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            print(
                f"Downloading huggingface file {repository_id}/{filename}"
                f"{f' via {endpoint}' if endpoint else ''}...",
                flush=True,
            )
            hf_hub_download(
                repo_id=repository_id,
                revision=revision,
                filename=filename,
                endpoint=endpoint,
                local_dir=local_directory,
                force_download=False,
            )
            if not verify_file(target_path, expected_sha256):
                raise ValueError(f"Downloaded file verification failed for {repository_id}/{filename}")
            return
        except Exception as exc:
            last_error = exc
            try:
                download_via_curl(repository_id, revision, filename, target_path, endpoint)
                if not verify_file(target_path, expected_sha256):
                    raise ValueError(f"curl fallback verification failed for {repository_id}/{filename}")
                return
            except Exception as curl_exc:
                last_error = curl_exc
            if attempt == retries:
                break
            wait_seconds = min(60, 5 * attempt)
            print(
                f"Download failed for {repository_id}/{filename} attempt {attempt}/{retries}: "
                f"{exc}; retrying in {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise last_error


def download_model(repository_id, files):
    local_directory = os.path.abspath(os.path.join("huggingface.co", repository_id))
    os.makedirs(local_directory, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT")
    revision = os.environ.get("RAGFLOW_HF_REVISION", "main")
    if endpoint:
        print(f"Using HF_ENDPOINT={endpoint} for {repository_id}", flush=True)
    retries = int(os.environ.get("RAGFLOW_HF_DOWNLOAD_RETRIES", "5"))
    for filename, expected_sha256 in files.items():
        download_hf_file(
            repository_id,
            revision,
            filename,
            expected_sha256,
            local_directory,
            endpoint,
            retries,
        )


def configure_china_mirrors():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("PIP_INDEX_URL", "https://mirrors.aliyun.com/pypi/simple")
    os.environ.setdefault("UV_INDEX_URL", "https://mirrors.aliyun.com/pypi/simple")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dependencies with optional China mirror support")
    parser.add_argument("--china-mirrors", action="store_true", help="Use China-accessible mirrors for downloads")
    args = parser.parse_args()

    if args.china_mirrors:
        configure_china_mirrors()

    if not env_flag("RAGFLOW_SKIP_URL_DOWNLOADS"):
        urls = get_urls(args.china_mirrors)

        for url in urls:
            candidates = url if isinstance(url, list) else [url]
            filename = candidates[1] if isinstance(url, list) and len(candidates) >= 2 else candidates[0].split("/")[-1]
            if os.path.exists(filename):
                continue
            # Earlier runs may have saved the chrome archives using the URL basename.
            # Reuse/rename that file so Dockerfile.deps can consume the expected alias.
            if isinstance(url, list) and len(candidates) >= 2:
                legacy_name = candidates[0].split("/")[-1]
                if legacy_name != filename and os.path.exists(legacy_name):
                    os.rename(legacy_name, filename)
                    continue
            last_error = None
            for download_url in candidates:
                if download_url == filename:
                    continue
                print(f"Downloading {filename} from {download_url}...", flush=True)
                try:
                    urllib.request.urlretrieve(download_url, filename)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"Download failed for {download_url}: {exc}", flush=True)
            if last_error is not None:
                raise last_error

    if not env_flag("RAGFLOW_SKIP_NLTK_DOWNLOADS"):
        import nltk

        local_dir = os.path.abspath("nltk_data")
        for data in ["wordnet", "punkt", "punkt_tab"]:
            print(f"Downloading nltk {data}...", flush=True)
            nltk.download(data, download_dir=local_dir)

    for repo_id, files in filtered_repo_files().items():
        print(f"Downloading huggingface repo {repo_id}...", flush=True)
        download_model(repo_id, files)
