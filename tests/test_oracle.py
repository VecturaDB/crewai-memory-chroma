"""The same random operations on LanceDBStorage, the default backend of CrewAI, and on ChromaStorage, with each result
compared. The operations avoid only the cases where the two backends differ on purpose, which tests/test_scopes.py covers:
no record in the root scope, no scope name that starts like another one, no scope without its leading slash (LanceDBStorage
adds it in some operations only), no delete of a scope with older_than. A search with categories is compared with what
QdrantEdgeStorage does (tests/test_categories.py), computed from LanceDBStorage. Every record has an embedding, so that no
two distances are the same.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import math
import os
import random
import tempfile

from crewai.memory.storage import lancedb_storage
from crewai.memory.types import MemoryRecord
import pytest


# The original class, also where tests/upstream/conftest.py has replaced it for the tests of CrewAI.
Reference = getattr(lancedb_storage, "_OriginalLanceDBStorage", lancedb_storage.LanceDBStorage)

DIM = 8
SCOPES = ["/a", "/a/b", "/a/c", "/b", "/b/a", "/c/a/b"]
QUERY_SCOPES = [*SCOPES, None, "/", "/c", "/c/a", "/a/"]
CATEGORIES = ["x", "y", "z"]
BASE = datetime(2026, 1, 1)
OPERATIONS = 120

# With CMC_STATS=<file>, each sequence appends to it how many operations of each kind it ran, and how many of them had a
# result that is not empty on LanceDBStorage: a kind never run, or never with a result, is a case the test does not prove.
_STATS_FILE = os.environ.get("CMC_STATS")


def _record(rng: random.Random, n: int) -> MemoryRecord:
    return MemoryRecord(
        id=f"r{n}",
        content=f"memory {n}",
        scope=rng.choice(SCOPES),
        categories=rng.sample(CATEGORIES, rng.randint(0, 2)),
        metadata={"k": rng.randint(0, 2)} if rng.random() < 0.7 else {},
        importance=round(rng.random(), 3),
        created_at=BASE + timedelta(seconds=rng.randint(0, 10**6), microseconds=rng.randint(1, 999_999)),
        last_accessed=BASE,
        embedding=[rng.gauss(0, 1) for _ in range(DIM)],
        source=rng.choice([None, "u1", "u2"]),
        private=rng.random() < 0.2,
    )


def _same_record(a: MemoryRecord | None, b: MemoryRecord | None, touched: bool, where: str) -> None:
    assert (a is None) == (b is None), where
    if a is None or b is None:
        return
    exclude = {"last_accessed"} if touched else set()
    assert a.model_dump(exclude=exclude) == b.model_dump(exclude=exclude), where
    assert a.embedding is not None and b.embedding is not None, where
    assert all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-6) for x, y in zip(a.embedding, b.embedding)), where


@pytest.mark.parametrize("seed", range(12))
def test_random_operations_give_the_results_of_lancedb(seed: int, new_storage) -> None:
    rng = random.Random(seed)
    reference = Reference(path=tempfile.mkdtemp(), vector_dim=DIM, compact_every=0)
    ours = new_storage(vector_dim=DIM)
    ids: list[str] = []
    n = 0
    touched = False
    stats: Counter[str] = Counter()
    for step in range(OPERATIONS):
        op = rng.choices(
            ["save", "search", "get", "list", "count", "info", "scopes", "categories", "delete", "update", "touch",
             "reset_scope", "reset_all"],
            weights=[25, 25, 5, 10, 5, 5, 5, 5, 10, 3, 1, 1, 0.5],
        )[0]
        where = f"seed {seed}, step {step}, {op}"
        stats[f"{op} run"] += 1
        if op == "save":
            batch = []
            for _ in range(rng.randint(1, 6)):
                n += 1
                batch.append(_record(rng, n))
            reference.save(batch)
            ours.save(batch)
            ids.extend(r.id for r in batch)
        elif op == "search":
            kwargs = {
                "scope_prefix": rng.choice(QUERY_SCOPES),
                "categories": rng.sample(CATEGORIES, rng.randint(1, 2)) if rng.random() < 0.3 else None,
                "metadata_filter": {"k": rng.randint(0, 2)} if rng.random() < 0.3 else None,
                "limit": rng.randint(1, 12),
                "min_score": rng.choice([0.0, 0.0, 0.02, 0.05]),
            }
            query = [rng.gauss(0, 1) for _ in range(DIM)]
            got_reference = reference.search(query, **kwargs)
            if kwargs["categories"]:
                # Categories are matched before the nearest records are taken, as QdrantEdgeStorage does: every record
                # of the categories in the scope, nearest first (LanceDBStorage with a limit above the size of the
                # table), then the nearest limit*3 with a metadata filter, or limit, then the metadata filter and
                # min_score, as LanceDBStorage does.
                pool = reference.search(query, scope_prefix=kwargs["scope_prefix"], categories=kwargs["categories"],
                                        limit=100_000)
                fetch = kwargs["limit"] * 3 if kwargs["metadata_filter"] else kwargs["limit"]
                expected = []
                for record, score in pool[:fetch]:
                    if kwargs["metadata_filter"] and not all(
                        record.metadata.get(k) == v for k, v in kwargs["metadata_filter"].items()
                    ):
                        continue
                    if score >= kwargs["min_score"]:
                        expected.append((record, score))
                    if len(expected) >= kwargs["limit"]:
                        break
                stats[f"{op} with categories"] += 1
                stats[f"{op} with categories, records LanceDB loses"] += [r.id for r, _ in expected] != [
                    r.id for r, _ in got_reference
                ]
                got_reference = expected
            got_ours = ours.search(query, **kwargs)
            stats[f"{op} with a result"] += bool(got_reference)
            assert [r.id for r, _ in got_ours] == [r.id for r, _ in got_reference], f"{where} {kwargs}"
            for (ra, sa), (rb, sb) in zip(got_reference, got_ours):
                assert math.isclose(sa, sb, rel_tol=1e-5), f"{where} {kwargs}"
                _same_record(ra, rb, touched, where)
        elif op == "get":
            record_id = rng.choice([*ids, "missing"]) if ids else "missing"
            got_reference = reference.get_record(record_id)
            stats[f"{op} with a result"] += got_reference is not None
            _same_record(got_reference, ours.get_record(record_id), touched, where)
        elif op == "list":
            kwargs = {"scope_prefix": rng.choice(QUERY_SCOPES), "limit": rng.randint(1, 30), "offset": rng.randint(0, 10)}
            got_reference = reference.list_records(**kwargs)
            got_ours = ours.list_records(**kwargs)
            stats[f"{op} with a result"] += bool(got_reference)
            assert [r.id for r in got_ours] == [r.id for r in got_reference], f"{where} {kwargs}"
            for a, b in zip(got_reference, got_ours):
                _same_record(a, b, touched, where)
        elif op == "count":
            scope = rng.choice(QUERY_SCOPES)
            expected = reference.count(scope)
            stats[f"{op} with a result"] += bool(expected)
            assert ours.count(scope) == expected, f"{where} {scope}"
        elif op == "info":
            scope = rng.choice([s for s in QUERY_SCOPES if s is not None])
            expected = reference.get_scope_info(scope)
            stats[f"{op} with a result"] += bool(expected.record_count and expected.child_scopes)
            assert ours.get_scope_info(scope) == expected, f"{where} {scope}"
        elif op == "scopes":
            parent = rng.choice(["/", "/a", "/b", "/c", "/c/a", "/a/"])
            expected = reference.list_scopes(parent)
            stats[f"{op} with a result"] += bool(expected)
            assert ours.list_scopes(parent) == expected, f"{where} {parent}"
        elif op == "categories":
            scope = rng.choice(QUERY_SCOPES)
            expected = reference.list_categories(scope)
            stats[f"{op} with a result"] += bool(expected)
            assert ours.list_categories(scope) == expected, f"{where} {scope}"
        elif op == "delete":
            kind = rng.choice(["ids", "ids_and_categories", "scope", "categories", "metadata", "older_than"])
            cutoff = BASE + timedelta(seconds=rng.randint(0, 10**6))
            kwargs: dict = {}
            if kind == "ids":
                kwargs = {"record_ids": [*rng.sample(ids, min(len(ids), rng.randint(1, 3))), "missing"]}
            elif kind == "ids_and_categories":
                kwargs = {"record_ids": rng.sample(ids, min(len(ids), 2)), "categories": [rng.choice(CATEGORIES)],
                          "scope_prefix": rng.choice(QUERY_SCOPES)}
            elif kind == "scope":
                kwargs = {"scope_prefix": rng.choice(SCOPES)}
            elif kind == "categories":
                kwargs = {"categories": [rng.choice(CATEGORIES)], "scope_prefix": rng.choice(QUERY_SCOPES),
                          "older_than": cutoff if rng.random() < 0.5 else None}
            elif kind == "metadata":
                kwargs = {"metadata_filter": {"k": rng.randint(0, 2)}, "scope_prefix": rng.choice(QUERY_SCOPES)}
            else:
                kwargs = {"older_than": cutoff}
            expected = reference.delete(**kwargs)
            stats[f"{op} {kind} run"] += 1
            stats[f"{op} {kind} with a result"] += bool(expected)
            assert ours.delete(**kwargs) == expected, f"{where} {kwargs}"
        elif op == "update":
            if not ids:
                continue
            original = reference.get_record(rng.choice(ids))
            if original is None:
                continue
            changed = original.model_copy(update={
                "content": original.content + " (updated)",
                "scope": rng.choice(SCOPES),
                "categories": rng.sample(CATEGORIES, rng.randint(0, 2)),
                "importance": round(rng.random(), 3),
                "embedding": [rng.gauss(0, 1) for _ in range(DIM)],
            })
            reference.update(changed)
            ours.update(changed)
        elif op == "touch":
            chosen = rng.sample(ids, min(len(ids), 3)) if ids else []
            reference.touch_records(chosen)
            ours.touch_records(chosen)
            touched = True
        elif op == "reset_scope":
            scope = rng.choice(SCOPES)
            reference.reset(scope)
            ours.reset(scope)
        else:
            reference.reset()
            ours.reset()
    assert ours.count() == reference.count()
    assert [r.id for r in ours.list_records(limit=10_000)] == [r.id for r in reference.list_records(limit=10_000)]
    if _STATS_FILE:
        with open(_STATS_FILE, "a", encoding="utf-8") as f:
            f.write(f"seed {seed}: {dict(sorted(stats.items()))}\n")
