"""The filters of ChromaStorage do not depend on a list in the metadata being absent, and reset() deletes the records
before the collection."""

from __future__ import annotations

from crewai.memory.types import MemoryRecord


def _raw_metadata(storage, record_id: str) -> dict:
    return storage._collection.get(ids=[record_id])["metadatas"][0]


def test_a_categories_list_written_by_another_client_gives_no_categories(new_storage) -> None:
    storage = new_storage(vector_dim=2)
    storage.save(
        [
            MemoryRecord(id="plain", content="plain", scope="/crew/a", embedding=[1.0, 0.0]),
            MemoryRecord(id="rare", content="rare", scope="/crew/a", categories=["rare"], embedding=[0.0, 1.0]),
        ]
    )
    storage._collection.update(ids=["plain"], metadatas=[{"categories": ["rare"]}])
    assert _raw_metadata(storage, "plain")["categories"] == ["rare"]
    assert storage.get_record("plain").categories == []
    # "plain" is the nearest record to the query: it must not take the only place of the result.
    assert [r.id for r, _ in storage.search([1.0, 0.0], categories=["rare"], limit=1)] == ["rare"]
    assert [r.id for r, _ in storage.search([1.0, 0.0], categories=["rare", "other"], limit=1)] == ["rare"]
    assert storage.list_categories() == {"rare": 1}
    assert storage.get_scope_info("/").categories == ["rare"]


def test_an_update_that_removes_every_category_removes_the_list(new_storage) -> None:
    storage = new_storage(vector_dim=2)
    record = MemoryRecord(id="r", content="r", scope="/crew/a", categories=["x", "y"], embedding=[1.0, 0.0])
    storage.save([record])
    storage.update(record.model_copy(update={"categories": []}))
    assert "categories" not in _raw_metadata(storage, "r")
    assert storage.get_record("r").categories == []
    assert storage.search([1.0, 0.0], categories=["x"], limit=5) == []


def test_reset_deletes_the_records_before_the_collection(new_storage, monkeypatch) -> None:
    storage = new_storage(vector_dim=2)
    storage.save([MemoryRecord(id=i, content=i, scope="/crew/a", embedding=[1.0, 0.0]) for i in ("a", "b", "c")])
    calls: list[tuple[str, object]] = []
    collection_type = type(storage._collection)
    delete_records = collection_type.delete
    delete_collection = storage._client.delete_collection

    def recording_delete(self, *args, **kwargs):
        calls.append(("records", sorted(kwargs.get("ids") or [])))
        return delete_records(self, *args, **kwargs)

    def recording_delete_collection(name):
        calls.append(("collection", name))
        return delete_collection(name)

    monkeypatch.setattr(collection_type, "delete", recording_delete)
    monkeypatch.setattr(storage._client, "delete_collection", recording_delete_collection)
    name = storage._collection_name
    storage.reset()
    assert calls == [("records", ["a", "b", "c"]), ("collection", name)]
    assert storage.count() == 0
