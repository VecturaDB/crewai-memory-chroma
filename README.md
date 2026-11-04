# crewai-memory-chroma

A storage backend for the memory of [CrewAI](https://github.com/crewAIInc/crewAI) that keeps the memories of a crew in a collection of a Chroma server, or of any server with the API of Chroma, such as [VecturaDB](https://github.com/VecturaDB/VecturaDB).

## Install

```bash
pip install crewai-memory-chroma
```

## Use

Start a server, for example:

```bash
docker run -d -p 8000:8000 chromadb/chroma:1.5.9
```

Then give the backend to the memory of the crew:

```python
import chromadb
from crewai import Crew, Memory
from crewai_memory_chroma import ChromaStorage

storage = ChromaStorage(
    client=chromadb.HttpClient(host="127.0.0.1", port=8000),
    collection_name="memories",
)
crew = Crew(agents=[...], tasks=[...], memory=Memory(storage=storage))
```

`client` takes any client of the `chromadb` package. Without it, `ChromaStorage` uses `chromadb.HttpClient()` with its defaults.

## What it does

It does what `LanceDBStorage`, the default backend of CrewAI, does: the same fields, the same scores (`1 / (1 + distance)`, with the squared L2 distance of Chroma), the same checks of the embedding dimension. Two things work as in `QdrantEdgeStorage`:

- Scopes are matched by path: `/crew/research` matches itself and `/crew/research/...`, not `/crew/research-2`.
- A search by category is filtered by the server before the nearest records are taken, so a record of that category is found even when other records are nearer.

## Tested with

In our tests, with CrewAI 1.15.35 and the `chromadb` client 1.5.9, against Chroma 1.5.9 and VecturaDB 1.0.0-preview.6:

- the memory tests of CrewAI 1.15.35 with `LanceDBStorage` replaced by this backend: 171 of 172 pass; the other one expects a search by a part of a scope name to match;
- random operations compared with `LanceDBStorage`, result by result, apart from the two differences above;
- the scopes, the categories, records without an embedding and the dimension checks.

## If CrewAI takes it in

We have proposed this backend to CrewAI in [crewAIInc/crewAI#9090](https://github.com/crewAIInc/crewAI/issues/9090). If CrewAI takes it in, this package will be deprecated and we will maintain the backend there.

## License

MIT. Provided as is, without warranty.

Independent project. Not affiliated with, endorsed by or sponsored by Chroma Inc. or crewAI, Inc. CHROMA is a registered trademark of Chroma Inc., used here only to describe compatibility.
