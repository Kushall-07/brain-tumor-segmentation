"""Tests for the path-safety helpers that guard every user-supplied path
before it reaches disk I/O or torch.load (see api/security.py)."""

import pytest

from api.security import UnsafePathError, resolve_within, resolve_within_any


def test_resolve_within_accepts_relative_path_inside_root(tmp_path):
    resolved = resolve_within(tmp_path, "sub/file.txt")
    assert resolved == (tmp_path / "sub" / "file.txt").resolve()


def test_resolve_within_accepts_absolute_path_inside_root(tmp_path):
    target = tmp_path / "file.txt"
    resolved = resolve_within(tmp_path, str(target))
    assert resolved == target.resolve()


def test_resolve_within_rejects_parent_traversal(tmp_path):
    with pytest.raises(UnsafePathError):
        resolve_within(tmp_path, "../escape.txt")


def test_resolve_within_rejects_nested_parent_traversal(tmp_path):
    with pytest.raises(UnsafePathError):
        resolve_within(tmp_path, "sub/../../escape.txt")


def test_resolve_within_rejects_absolute_path_outside_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    other = tmp_path_factory.mktemp("other")
    with pytest.raises(UnsafePathError):
        resolve_within(root, str(other / "secret.txt"))


def test_resolve_within_any_resolves_against_first_matching_root(tmp_path_factory):
    root_a = tmp_path_factory.mktemp("a")
    root_b = tmp_path_factory.mktemp("b")
    resolved = resolve_within_any((root_a, root_b), "file.txt")
    assert resolved == (root_a / "file.txt").resolve()


def test_resolve_within_any_resolves_against_second_root_when_absolute(tmp_path_factory):
    root_a = tmp_path_factory.mktemp("a")
    root_b = tmp_path_factory.mktemp("b")
    target = root_b / "file.txt"
    resolved = resolve_within_any((root_a, root_b), str(target))
    assert resolved == target.resolve()


def test_resolve_within_any_rejects_when_outside_all_roots(tmp_path_factory):
    root_a = tmp_path_factory.mktemp("a")
    root_b = tmp_path_factory.mktemp("b")
    outside = tmp_path_factory.mktemp("outside")
    with pytest.raises(UnsafePathError):
        resolve_within_any((root_a, root_b), str(outside / "secret.txt"))
