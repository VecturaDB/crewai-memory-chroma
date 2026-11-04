"""A record saved without an embedding is kept with a vector of zeros, as LanceDBStorage keeps it: get_record returns the
zeros, a search finds it at the distance of the query from the origin, and the dimension of the store stays the one of the
records with an embedding.
"""

from __future__ import annotations

import tempfile

from crewai.memory.storage import lancedb_storage
from crewai.memory.types import MemoryRecord
import pytest


Reference = getattr(lancedb_storage, "_OriginalLanceDBStorage", lancedb_storage.LanceDBStorage)


@pytest.mark.parametrize("embedding", [None, []], ids=["none", "empty"])
def test_a_record_without_embedding_is_kept_with_zeros_as_lancedb_keeps_it(new_storage, embedding) -> None:
    reference = Reference(path=tempfile.mkdtemp(), vector_dim=3, compact_every=0)
    ours = new_storage(vector_dim=3)
    records = [
        MemoryRecord(id="with", content="with", scope="/s", embedding=[1.0, 2.0, 2.0]),
        MemoryRecord(id="without", content="without", scope="/s", embedding=embedding),
    ]
    for storage in (reference, ours):
        storage.save(records)
    assert ours.get_record("without").embedding == reference.get_record("without").embedding == [0.0, 0.0, 0.0]
    query = [0.0, 3.0, 4.0]
    got_reference = [(r.id, round(s, 6)) for r, s in reference.search(query, limit=5)]
    got_ours = [(r.id, round(s, 6)) for r, s in ours.search(query, limit=5)]
    assert got_ours == got_reference
    assert ("without", round(1 / (1 + 25.0), 6)) in got_ours
    assert ours.count() == reference.count() == 2


def test_a_first_save_without_any_embedding_takes_the_default_dimension(new_storage) -> None:
    reference = Reference(path=tempfile.mkdtemp(), compact_every=0)
    ours = new_storage()
    for storage in (reference, ours):
        storage.save([MemoryRecord(id="without", content="without", scope="/s")])
    assert len(ours.get_record("without").embedding) == len(reference.get_record("without").embedding) == 3072
