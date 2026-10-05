"""Path-safety helpers for user-supplied filesystem paths.

Any path that originates from an HTTP request (query param, JSON body field,
etc.) must be validated with these helpers before it is used to read, write,
or load a file. This is the single place that decides whether a candidate
path is allowed to escape the directory it is supposed to stay inside of.
"""

from __future__ import annotations

from pathlib import Path


class UnsafePathError(ValueError):
    """Raised when a user-supplied path escapes an allowed root directory."""


def resolve_within(root: Path, candidate: str | Path) -> Path:
    """Resolve `candidate` and verify it stays inside `root`.

    Absolute candidates are resolved as-is; relative candidates are joined
    onto `root` first. Either way, the final resolved path must be `root`
    itself or a descendant of it, otherwise `UnsafePathError` is raised
    (this blocks `..` traversal and absolute paths pointing elsewhere).
    """
    root_resolved = root.resolve()
    candidate_path = Path(candidate)
    resolved = (
        candidate_path.resolve()
        if candidate_path.is_absolute()
        else (root_resolved / candidate_path).resolve()
    )

    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise UnsafePathError(
            f"Path '{candidate}' is outside allowed directory '{root_resolved}'"
        ) from exc

    return resolved


def resolve_within_any(roots: tuple[Path, ...], candidate: str | Path) -> Path:
    """Resolve `candidate` against the first root in `roots` it fits inside of."""
    last_error: Exception | None = None
    for root in roots:
        try:
            return resolve_within(root, candidate)
        except UnsafePathError as exc:
            last_error = exc
    raise UnsafePathError(
        f"Path '{candidate}' is outside all allowed directories: "
        f"{[str(r) for r in roots]}"
    ) from last_error
