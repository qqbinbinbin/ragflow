from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _dependency_downloader_source() -> str:
    candidates = [
        ROOT / "download_deps.py",
        ROOT / "ragflow_deps" / "download_deps.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise AssertionError("missing maintained dependency downloader")


def test_dependency_downloader_keeps_bounded_offline_controls():
    source = _dependency_downloader_source()

    assert source.index(
        'os.environ.setdefault("HF_HUB_DISABLE_XET"'
    ) < source.index("from huggingface_hub import")
    assert "RAGFLOW_SKIP_URL_DOWNLOADS" in source
    assert "RAGFLOW_SKIP_NLTK_DOWNLOADS" in source
    assert "RAGFLOW_HF_ONLY_REPOS" in source
    assert "RAGFLOW_HF_ONLY_FILES" in source
    assert "RAGFLOW_HF_DOWNLOAD_RETRIES" in source
    assert "compute_sha256" in source
    assert "verify_file" in source


def test_image_build_keeps_configurable_china_mirror_and_offline_git_source():
    source = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "ARG APT_MIRROR=" in source
    assert "ARG UV_INDEX_URL=" in source
    assert ".offline-cache/graspologic.git" in source
    assert 'insteadOf "https://github.com/infiniflow/graspologic.git"' in source
