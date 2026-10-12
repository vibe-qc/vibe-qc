"""Filesystem isolation checks without loading the native chemistry engine."""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _cache_selector():
    source = ROOT / "python/vibeqc/basis_filter.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_filtered_basis_cache_dir")
    namespace = {"os": os, "Path": Path}
    exec(compile(ast.Module([node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[node.name]


class CacheIsolationTests(unittest.TestCase):
    def test_writable_installed_library_is_not_the_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            installed = base / "installed"
            (installed / "basis").mkdir(parents=True)
            stock = installed / "basis/sto-3g.g94"
            stock.write_text("synthetic stock basis\n")
            with patch.dict(os.environ, {
                "LIBINT_DATA_PATH": str(installed),
                "XDG_CACHE_HOME": str(base / "cache"),
            }):
                os.environ.pop("VIBEQC_FILTERED_BASIS_DIR", None)
                target = _cache_selector()()
                self.assertFalse(target.resolve().is_relative_to(installed.resolve()),
                                 "filtered files would be written into the installed library")
            self.assertEqual(stock.read_text(), "synthetic stock basis\n")


def _store():
    path = ROOT / "python/vibeqc/_filtered_basis_store.py"
    spec = importlib.util.spec_from_file_location("filtered_store_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "dont_write_bytecode", True):
        spec.loader.exec_module(module)
    return module


class MaterializationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "installed"
        (self.source / "basis").mkdir(parents=True)
        (self.source / "basis/stock.g94").write_bytes(b"stock basis bytes")
        (self.source / "basis/stock.ecp").write_bytes(b"stock sidecar bytes")
        self.cache = self.base / "cache"
        self.store = _store()

    def build(self, source=None, name="_vibeqc_filtered_one.g94", text="derived basis"):
        return self.store.materialize(self.cache, source or self.source, name, text)

    def test_root_is_a_complete_copy_with_content_identity(self):
        root = self.build()
        self.assertEqual(self.store.validate_root(root)["stock.ecp"], b"stock sidecar bytes")
        self.assertFalse(any(p.is_symlink() for p in (root / "basis").iterdir()))
        self.assertEqual(sorted(p.name for p in (self.source / "basis").iterdir()), ["stock.ecp", "stock.g94"])
        self.assertEqual(self.build(), root)
        self.assertEqual(len(list(self.cache.iterdir())), 1)

    def test_previous_names_survive_later_materialization(self):
        first = self.build()
        second = self.build(first, "_vibeqc_filtered_two.g94", "second")
        self.assertEqual((second / "basis/_vibeqc_filtered_one.g94").read_bytes(), b"derived basis")
        self.assertEqual((first / "basis/_vibeqc_filtered_one.g94").read_bytes(), b"derived basis")
        self.assertTrue((second / "basis/stock.g94").is_file())

    def test_changed_cached_bytes_refuse_reuse_and_derivation(self):
        root = self.build()
        (root / "basis/stock.g94").write_bytes(b"corrupted bytes")
        for source in (self.source, root):
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                self.build(source)

    def test_missing_extra_and_linked_cached_files_refuse(self):
        for mutation in ("missing", "extra", "symlink"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                self.cache = Path(directory)
                root = self.build()
                if mutation == "missing":
                    (root / "basis/stock.ecp").unlink()
                elif mutation == "extra":
                    (root / "basis/extra.g94").write_bytes(b"extra")
                else:
                    (root / "basis/stock.ecp").unlink()
                    (root / "basis/stock.ecp").symlink_to(self.source / "basis/stock.ecp")
                with self.assertRaises(RuntimeError):
                    self.build()

    def test_source_file_links_are_copied_without_writing_through(self):
        target = self.source / "basis/stock.g94"
        link = self.source / "basis/link.g94"
        link.symlink_to(target)
        root = self.build()
        self.assertEqual((root / "basis/link.g94").read_bytes(), target.read_bytes())
        self.assertFalse((root / "basis/link.g94").is_symlink())
        self.assertTrue(link.is_symlink())

    def test_name_collision_and_casefold_collision_refuse(self):
        first = self.build()
        with self.assertRaisesRegex(RuntimeError, "conflicts"):
            self.build(first, text="different derived basis")
        # Model a case-sensitive source listing on every host filesystem.
        entries = [self.source / "basis/stock.g94", self.source / "basis/STOCK.g94"]
        with patch.object(Path, "iterdir", return_value=iter(entries)):
            with self.assertRaisesRegex(RuntimeError, "case-fold collision"):
                self.build()

    def test_installed_directory_and_symlink_destination_refuse(self):
        for cache in (self.source / "cache", self.base / "link"):
            if cache.name == "link":
                cache.symlink_to(self.source, target_is_directory=True)
            self.cache = cache
            with self.assertRaisesRegex(RuntimeError, "outside"):
                self.build()
        self.assertEqual(sorted(p.name for p in (self.source / "basis").iterdir()), ["stock.ecp", "stock.g94"])

    def test_concurrent_publication_converges_without_partial_roots(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            roots = list(executor.map(lambda unused: self.build(), range(2)))
        self.assertEqual(roots[0], roots[1])
        self.store.validate_root(roots[0])
        self.assertEqual([p.name for p in self.cache.iterdir()], [roots[0].name])

    def test_explicit_purge_refuses_selected_root_then_returns_real_count(self):
        root = self.build()
        with self.assertRaisesRegex(RuntimeError, "selected"):
            self.store.purge(self.cache, root)
        self.assertTrue(root.is_dir())
        self.assertEqual(self.store.purge(self.cache, self.source), 1)
        self.assertEqual(self.store.purge(self.cache, self.source), 0)

    def test_purge_preflights_all_roots_and_refuses_unknown_or_linked_entries(self):
        root = self.build()
        unknown = self.cache / "unknown"
        unknown.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(RuntimeError):
            self.store.purge(self.cache, self.source)
        self.assertTrue(root.is_dir())
        self.assertTrue((self.source / "basis/stock.g94").is_file())

    def test_declared_byte_bound_refuses_without_large_allocation(self):
        self.store._MAX_LIBRARY_BYTES = 4
        with self.assertRaisesRegex(RuntimeError, "byte bound"):
            self.build()
        self.assertFalse(self.cache.exists())

    def test_missing_receipt_never_downgrades_cache_to_an_unchecked_source(self):
        root = self.build()
        (root / "filtered-basis-receipt.json").unlink()
        with self.assertRaises(RuntimeError):
            self.build(root, "_vibeqc_filtered_two.g94", "second")

    def test_direct_libint_basis_directory_preserves_prior_cache_validation(self):
        first = self.build()
        second = self.build(first / "basis", "_vibeqc_filtered_two.g94", "second")
        self.assertEqual((second / "basis/stock.g94").read_bytes(), b"stock basis bytes")
        self.assertTrue((second / "basis/_vibeqc_filtered_one.g94").is_file())
