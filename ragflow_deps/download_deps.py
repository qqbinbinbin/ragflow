#!/usr/bin/env python3

# PEP 723 metadata
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "nltk",
#   "huggingface-hub"
# ]
# ///

# This script downloads every artifact that the `infiniflow/ragflow_deps`
# Docker image bakes in. Run it from anywhere — the `__main__` block
# chdir's into this file's own directory, so all outputs land under
# `ragflow_deps/` regardless of the caller's CWD.
#
# Build-context relationship: `ragflow_deps/Dockerfile` is built with
# `ragflow_deps/` as its build context, so the files written here MUST
# sit at the top of `ragflow_deps/`. The Dockerfile's COPY lines assume
# top-level paths (`huggingface.co`, `nltk_data`, `cl100k_base.tiktoken`,
# `*.deb`, `*.jar`, `*.tar.gz`, `stagehand-server-v3-linux-<arch>`).
#
# Typical workflow:
#
#   uv run python3 ragflow_deps/download_deps.py            # download
#   cd ragflow_deps
#   docker build -f Dockerfile -t infiniflow/ragflow_deps .
#
# The main `Dockerfile` (built from the project root) pulls this image
# via `--mount=type=bind,from=infiniflow/ragflow_deps:latest,...` and
# is unaffected by where these files live locally.

import argparse
import hashlib
import os
import subprocess
import time
import urllib.request
from typing import Union

# huggingface_hub reads these at import time.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from huggingface_hub import hf_hub_download, snapshot_download


ALIYUN_UV_MIRROR = "https://mirrors.aliyun.com/github-release/astral-sh/uv/0.9.16/"
GITHUB_UV_RELEASE = "https://github.com/astral-sh/uv/releases/download/0.9.16/"


def get_urls(use_china_mirrors=False) -> list[Union[str, list[str]]]:
    if use_china_mirrors:
        return [
            "http://mirrors.tuna.tsinghua.edu.cn/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb",
            "http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_arm64.deb",
            "https://repo.huaweicloud.com/repository/maven/org/apache/tika/tika-server-standard/3.3.0/tika-server-standard-3.3.0.jar",
            "https://repo.huaweicloud.com/repository/maven/org/apache/tika/tika-server-standard/3.3.0/tika-server-standard-3.3.0.jar.md5",
            "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
            ["https://registry.npmmirror.com/-/binary/chrome-for-testing/121.0.6167.85/linux64/chrome-linux64.zip", "chrome-linux64-121-0-6167-85"],
            ["https://registry.npmmirror.com/-/binary/chrome-for-testing/121.0.6167.85/linux64/chromedriver-linux64.zip", "chromedriver-linux64-121-0-6167-85"],
            [f"{ALIYUN_UV_MIRROR}uv-x86_64-unknown-linux-gnu.tar.gz", "uv-x86_64-unknown-linux-gnu.tar.gz", f"{GITHUB_UV_RELEASE}uv-x86_64-unknown-linux-gnu.tar.gz"],
            [f"{ALIYUN_UV_MIRROR}uv-aarch64-unknown-linux-gnu.tar.gz", "uv-aarch64-unknown-linux-gnu.tar.gz", f"{GITHUB_UV_RELEASE}uv-aarch64-unknown-linux-gnu.tar.gz"],
            # stagehand-server-v3 Node.js SEA binaries (used by Browser
            # component in local mode).
            #
            # The stagehand-go Go module (pinned in go.mod) and the
            # stagehand-server binary (this release) are LOOSELY
            # MATCHED — both stay on the v3.x line and remain
            # protocol-compatible. The two version numbers do NOT
            # track each other: the Go SDK is at v3.21.0 while the
            # current latest server release is v3.7.2.
            #
            # On every go.mod bump, refresh this URL to the current
            # latest server release. There is no version
            # correspondence to maintain; "both on v3.x" is the
            # compatibility contract.
            "https://github.com/browserbase/stagehand/releases/download/stagehand-server-v3/v3.7.2/stagehand-server-v3-linux-x64",
            "https://github.com/browserbase/stagehand/releases/download/stagehand-server-v3/v3.7.2/stagehand-server-v3-linux-arm64",
            # Native static libraries for Go build (pdfium, pdf_oxide, office_oxide)
            # Used by build.sh's check_*_deps functions — pre-downloaded to avoid
            # network access during CI.
            ["https://github.com/kognitos/pdfium-static/releases/download/chromium%2F7809/pdfium-linux-x64-static.tgz", "pdfium-linux-x64-static.tgz"],
            ["https://github.com/yfedoseev/pdf_oxide/releases/download/v0.3.67/pdf_oxide-go-ffi-linux-amd64.tar.gz", "pdf_oxide-go-ffi-linux-amd64.tar.gz"],
            ["https://github.com/yfedoseev/office_oxide/releases/download/v0.1.8/native-linux-x86_64.tar.gz", "office_oxide-linux-x86_64.tar.gz"],
        ]
    else:
        return [
            "http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb",
            "http://ports.ubuntu.com/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_arm64.deb",
            "https://repo1.maven.org/maven2/org/apache/tika/tika-server-standard/3.3.0/tika-server-standard-3.3.0.jar",
            "https://repo1.maven.org/maven2/org/apache/tika/tika-server-standard/3.3.0/tika-server-standard-3.3.0.jar.md5",
            "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
            ["https://storage.googleapis.com/chrome-for-testing-public/121.0.6167.85/linux64/chrome-linux64.zip", "chrome-linux64-121-0-6167-85"],
            ["https://storage.googleapis.com/chrome-for-testing-public/121.0.6167.85/linux64/chromedriver-linux64.zip", "chromedriver-linux64-121-0-6167-85"],
            "https://github.com/astral-sh/uv/releases/download/0.9.16/uv-x86_64-unknown-linux-gnu.tar.gz",
            "https://github.com/astral-sh/uv/releases/download/0.9.16/uv-aarch64-unknown-linux-gnu.tar.gz",
            # stagehand-server-v3 Node.js SEA binaries (used by Browser
            # component in local mode).
            #
            # The stagehand-go Go module (pinned in go.mod) and the
            # stagehand-server binary (this release) are LOOSELY
            # MATCHED — both stay on the v3.x line and remain
            # protocol-compatible. The two version numbers do NOT
            # track each other: the Go SDK is at v3.21.0 while the
            # current latest server release is v3.7.2.
            #
            # On every go.mod bump, refresh this URL to the current
            # latest server release. There is no version
            # correspondence to maintain; "both on v3.x" is the
            # compatibility contract.
            "https://github.com/browserbase/stagehand/releases/download/stagehand-server-v3/v3.7.2/stagehand-server-v3-linux-x64",
            "https://github.com/browserbase/stagehand/releases/download/stagehand-server-v3/v3.7.2/stagehand-server-v3-linux-arm64",
            # Native static libraries for Go build (pdfium, pdf_oxide, office_oxide)
            # Used by build.sh's check_*_deps functions — pre-downloaded to avoid
            # network access during CI.
            ["https://github.com/kognitos/pdfium-static/releases/download/chromium%2F7809/pdfium-linux-x64-static.tgz", "pdfium-linux-x64-static.tgz"],
            ["https://github.com/yfedoseev/pdf_oxide/releases/download/v0.3.67/pdf_oxide-go-ffi-linux-amd64.tar.gz", "pdf_oxide-go-ffi-linux-amd64.tar.gz"],
            ["https://github.com/yfedoseev/office_oxide/releases/download/v0.1.8/native-linux-x86_64.tar.gz", "office_oxide-linux-x86_64.tar.gz"],
        ]


repos = [
    "InfiniFlow/text_concat_xgb_v1.0",
    "InfiniFlow/deepdoc",
]


repo_files = {
    "InfiniFlow/text_concat_xgb_v1.0": {"updown_concat_xgb.model": None},
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
    return {item.strip() for item in os.environ.get(name, "").split(",") if item.strip()}


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


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
    return compute_sha256(path) == expected_sha256


def download_hf_file(repository_id, filename, local_directory, endpoint, revision):
    target_path = os.path.join(local_directory, filename)
    hf_hub_download(
        repo_id=repository_id,
        revision=revision,
        filename=filename,
        endpoint=endpoint,
        local_dir=local_directory,
        force_download=False,
    )
    expected_sha256 = repo_files.get(repository_id, {}).get(filename)
    if not verify_file(target_path, expected_sha256):
        raise ValueError(f"Downloaded file verification failed for {repository_id}/{filename}")


def download_model(repository_id):
    local_directory = os.path.abspath(os.path.join("huggingface.co", repository_id))
    os.makedirs(local_directory, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT")
    revision = os.environ.get("RAGFLOW_HF_REVISION", "main")
    only_files = parse_csv_env("RAGFLOW_HF_ONLY_FILES")
    retries = max(1, int(os.environ.get("RAGFLOW_HF_DOWNLOAD_RETRIES", "5")))
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            if only_files:
                for filename in sorted(only_files):
                    download_hf_file(repository_id, filename, local_directory, endpoint, revision)
            else:
                snapshot_download(repo_id=repository_id, revision=revision, endpoint=endpoint, local_dir=local_directory)
                for filename, expected_sha256 in repo_files.get(repository_id, {}).items():
                    if expected_sha256 and not verify_file(os.path.join(local_directory, filename), expected_sha256):
                        raise ValueError(f"Downloaded file verification failed for {repository_id}/{filename}")
            return
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(60, 5 * attempt))
    raise last_error


if __name__ == "__main__":
    # Anchor CWD to this file's directory so all relative outputs
    # (huggingface.co/, nltk_data/, *.deb, *.jar, *.tar.gz, etc.) land
    # at the top of ragflow_deps/ regardless of where the user invokes
    # the script from. This is the build context for `ragflow_deps/Dockerfile`.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Download dependencies with optional China mirror support")
    parser.add_argument("--china-mirrors", action="store_true", help="Use China-accessible mirrors for downloads")
    args = parser.parse_args()

    if args.china_mirrors:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("PIP_INDEX_URL", "https://mirrors.aliyun.com/pypi/simple")
        os.environ.setdefault("UV_INDEX_URL", "https://mirrors.aliyun.com/pypi/simple")

    urls = [] if env_flag("RAGFLOW_SKIP_URL_DOWNLOADS") else get_urls(args.china_mirrors)

    # Some mirrors (e.g. archive.ubuntu.com) reject the default urllib
    # User-Agent with HTTP 403, so install an opener with a browser-like UA.
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)

    for url in urls:
        candidates = url if isinstance(url, list) else [url]
        filename = candidates[1] if isinstance(url, list) and len(candidates) >= 2 else candidates[0].split("/")[-1]
        if os.path.exists(filename):
            continue
        last_error = None
        for download_url in candidates:
            if download_url == filename:
                continue
            try:
                print(f"Downloading {filename} from {download_url}...", flush=True)
                urllib.request.urlretrieve(download_url, filename)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error

    # Extract native static libraries to ~/ragflow-native-libs for Go build.
    # Ensures build.sh can find them without network access.
    native_deps_dir = os.path.expanduser("~/ragflow-native-libs")
    extractions = [
        ("pdfium-linux-x64-static.tgz", "pdfium-static"),
        ("pdf_oxide-go-ffi-linux-amd64.tar.gz", "pdf_oxide"),
        ("office_oxide-linux-x86_64.tar.gz", "office_oxide"),
    ]
    import tarfile

    for archive, subdir in extractions:
        archive_path = os.path.join(os.getcwd(), archive)
        if not os.path.isfile(archive_path):
            print(f"  Skipping extraction: {archive} not found")
            continue
        target = os.path.join(native_deps_dir, subdir)
        if os.path.isdir(target):
            print(f"  ✓ {subdir} already extracted to {target}")
            continue
        os.makedirs(target, exist_ok=True)
        print(f"  Extracting {archive} → {target}")
        with tarfile.open(archive_path) as tf:
            tf.extractall(target)

    if not env_flag("RAGFLOW_SKIP_NLTK_DOWNLOADS"):
        import nltk

        local_dir = os.path.abspath("nltk_data")
        for data in ["wordnet", "punkt", "punkt_tab"]:
            print(f"Downloading nltk {data}...")
            nltk.download(data, download_dir=local_dir)

    only_repos = parse_csv_env("RAGFLOW_HF_ONLY_REPOS")
    for repo_id in (repo for repo in repos if not only_repos or repo in only_repos):
        print(f"Downloading huggingface repo {repo_id}...")
        download_model(repo_id)
