"""Shared year-based storage paths for downloads and prepared GDAS inputs."""

from pathlib import Path


def raw_path(root, init_time):
    from .download import build_filename

    return Path(root) / str(init_time.year) / build_filename(init_time)


def existing_raw_path(root, init_time):
    """Read flat legacy archives without downloading another copy."""
    path = raw_path(root, init_time)
    legacy = Path(root) / path.name
    return path if path.exists() or not legacy.exists() else legacy


def prepared_directory(root, init_time):
    return Path(root) / str(init_time.year) / init_time.strftime("%Y-%m-%d-%H-%M")


def input_directory(root, init_time):
    directory = prepared_directory(root, init_time)
    legacy = Path(root) / init_time.strftime("%Y-%m-%d-%H-%M")
    return directory if directory.exists() or not legacy.exists() else legacy

