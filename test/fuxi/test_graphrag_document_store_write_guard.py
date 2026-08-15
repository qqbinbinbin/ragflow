import ast
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_PATH = REPO_ROOT / "rag" / "graphrag" / "utils.py"
INDEX_PATH = REPO_ROOT / "rag" / "graphrag" / "general" / "index.py"


def _load_functions(path: Path, names: set[str], namespace: dict):
    module = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


async def _run_sync(function, *args, **kwargs):
    return function(*args, **kwargs)


class _GuardedStore:
    def __init__(
        self,
        guard_state,
        *,
        existing_fields=None,
        existing_pages=None,
        search_error=None,
    ):
        self.guard_state = guard_state
        self.existing_fields = existing_fields or {}
        self.existing_pages = existing_pages
        self.search_error = search_error
        self.events = []
        self.search_offsets = []

    def search(self, *_args, **_kwargs):
        if self.search_error is not None:
            raise self.search_error
        offset = _args[5]
        self.search_offsets.append(offset)
        return SimpleNamespace(offset=offset)

    def get_fields(self, result, _fields):
        if self.existing_pages is not None:
            return self.existing_pages.get(result.offset, {})
        return self.existing_fields

    def delete(self, condition, index_name, dataset_id):
        assert self.guard_state["active"] is True
        self.events.append(("delete", condition, index_name, dataset_id))

    def insert(self, chunks, index_name, dataset_id):
        assert self.guard_state["active"] is True
        self.events.append(("insert", chunks, index_name, dataset_id))
        return None


class _RecordingDocumentService:
    def __init__(self, guard_state, *, reject=False):
        self.guard_state = guard_state
        self.reject = reject
        self.calls = []

    def execute_document_store_write(
        self,
        document_ids,
        dataset_id,
        write_operation,
        *args,
        **kwargs,
    ):
        normalized_ids = tuple(document_ids)
        self.calls.append((normalized_ids, dataset_id))
        if self.reject:
            raise RuntimeError("document is canceled for deletion")
        assert self.guard_state["active"] is False
        self.guard_state["active"] = True
        try:
            return write_operation(*args, **kwargs)
        finally:
            self.guard_state["active"] = False


class _Nodes(dict):
    def __call__(self):
        return list(self)


class _MiniGraph:
    def __init__(self):
        self.graph = {}
        self.nodes = _Nodes()
        self._edges = {}

    def add_node(self, name, **attributes):
        self.nodes[name] = attributes

    def add_edge(self, source, target, **attributes):
        self._edges[(source, target)] = attributes

    def has_node(self, name):
        return name in self.nodes

    def get_edge_data(self, source, target):
        return self._edges.get((source, target))

    def edges(self):
        return list(self._edges)

    def subgraph(self, names):
        selected = set(names)
        graph = _MiniGraph()
        graph.nodes.update(
            (name, dict(attributes))
            for name, attributes in self.nodes.items()
            if name in selected
        )
        graph._edges.update(
            (edge, dict(attributes))
            for edge, attributes in self._edges.items()
            if edge[0] in selected and edge[1] in selected
        )
        return graph

    def copy(self):
        graph = _MiniGraph()
        graph.graph = dict(self.graph)
        graph.nodes.update(
            (name, dict(attributes))
            for name, attributes in self.nodes.items()
        )
        graph._edges.update(
            (edge, dict(attributes))
            for edge, attributes in self._edges.items()
        )
        return graph


class _NetworkX:
    Graph = _MiniGraph

    @staticmethod
    def node_link_data(graph, **_kwargs):
        return {
            "graph": graph.graph,
            "nodes": list(graph.nodes),
            "edges": list(graph._edges),
        }


class _AsyncLimiter:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


def _utils_module(store, document_service):
    namespace = {
        "asyncio": asyncio,
        "DocumentService": document_service,
        "logging": logging,
        "os": os,
        "time": time,
        "thread_pool_exec": _run_sync,
        "settings": SimpleNamespace(docStoreConn=store),
        "search": SimpleNamespace(index_name=lambda tenant_id: f"index-{tenant_id}"),
        "_INSERT_BULK_SIZE": 2,
        "_INSERT_CONCURRENCY": 1,
        "_batch_embed_cache_misses": lambda *_args: [],
        "_write_embed_cache_batch": lambda *_args: None,
        "get_uuid": lambda: "graph-chunk",
        "nx": _NetworkX,
        "json": json,
        "chat_limiter": _AsyncLimiter(),
        "graph_node_to_chunk": None,
        "graph_edge_to_chunk": None,
        "GraphChange": object,
        "OrderByExpr": lambda: object(),
    }
    return _load_functions(
        UTILS_PATH,
        {
            "_source_document_ids",
            "_insert_chunks_sync",
            "_insert_chunks_with_retry_sync",
            "_load_graph_write_source_ids",
            "insert_chunks_bounded",
            "set_graph",
        },
        namespace,
    )


def _index_module(store, document_service, utils_module):
    async def false_async(*_args, **_kwargs):
        return False

    async def cleanup_async(*_args, **_kwargs):
        return None

    async def empty_checkpoints(*_args, **_kwargs):
        return {}

    namespace = {
        "asyncio": asyncio,
        "DocumentService": document_service,
        "CommunityReportsExtractor": None,
        "Extractor": object,
        "OrderByExpr": lambda: object(),
        "cleanup_checkpoints": cleanup_async,
        "COMMUNITY_CHECKPOINT": "community",
        "does_graph_contains": false_async,
        "json": json,
        "logging": logging,
        "nx": _NetworkX,
        "settings": SimpleNamespace(docStoreConn=store),
        "search": SimpleNamespace(index_name=lambda tenant_id: f"index-{tenant_id}"),
        "thread_pool_exec": _run_sync,
        "timeout": lambda *_args, **_kwargs: lambda function: function,
        "_has_cancel_and_exit": lambda *_args, **_kwargs: None,
        "tidy_graph": lambda *_args, **_kwargs: None,
        "chunk_id": lambda _chunk: "graph-chunk",
        "rag_tokenizer": SimpleNamespace(
            tokenize=lambda value: value,
            fine_grained_tokenize=lambda value: value,
        ),
        "load_checkpoints": empty_checkpoints,
        "save_checkpoint": cleanup_async,
        "insert_chunks_bounded": utils_module.insert_chunks_bounded,
        "_insert_chunks_sync": utils_module._insert_chunks_sync,
        "_insert_chunks_with_retry_sync": (
            utils_module._insert_chunks_with_retry_sync
        ),
        "_source_document_ids": getattr(utils_module, "_source_document_ids", None),
    }
    return _load_functions(
        INDEX_PATH,
        {"generate_subgraph", "extract_community"},
        namespace,
    )


@pytest.mark.asyncio
async def test_bounded_insert_guards_each_batch_with_exact_source_document_union():
    guard_state = {"active": False}
    store = _GuardedStore(guard_state)
    document_service = _RecordingDocumentService(guard_state)
    module = _utils_module(store, document_service)

    await module.insert_chunks_bounded(
        [
            {"id": "chunk-1", "source_id": ["doc-b", "doc-a"]},
            {"id": "chunk-2", "source_id": "doc-b"},
            {"id": "chunk-3", "source_id": ["doc-c", "doc-a"]},
        ],
        "tenant-1",
        "dataset-1",
    )

    assert document_service.calls == [
        (("doc-a", "doc-b"), "dataset-1"),
        (("doc-a", "doc-c"), "dataset-1"),
    ]
    assert [event[0] for event in store.events] == ["insert", "insert"]


@pytest.mark.asyncio
async def test_bounded_insert_does_not_retry_a_rejected_document_guard():
    guard_state = {"active": False}
    store = _GuardedStore(guard_state)
    document_service = _RecordingDocumentService(guard_state, reject=True)
    module = _utils_module(store, document_service)

    with pytest.raises(RuntimeError, match="canceled for deletion"):
        await module.insert_chunks_bounded(
            [{"id": "chunk-1", "source_id": ["doc-deleting"]}],
            "tenant-1",
            "dataset-1",
        )

    assert document_service.calls == [
        (("doc-deleting",), "dataset-1"),
    ]
    assert store.events == []


def test_source_document_union_rejects_any_unscoped_chunk():
    guard_state = {"active": False}
    store = _GuardedStore(guard_state)
    document_service = _RecordingDocumentService(guard_state)
    module = _utils_module(store, document_service)

    with pytest.raises(ValueError, match="requires a source document"):
        module._source_document_ids(
            [
                {"id": "scoped", "source_id": ["doc-1"]},
                {"id": "unscoped", "source_id": []},
            ]
        )


def test_existing_graph_source_union_reads_all_pages_without_truncation():
    guard_state = {"active": False}
    first_page = {
        f"subgraph-{index}": {"source_id": [f"doc-{index}"]}
        for index in range(1000)
    }
    store = _GuardedStore(
        guard_state,
        existing_pages={
            0: first_page,
            1000: {"last-subgraph": {"source_id": ["doc-last"]}},
        },
    )
    document_service = _RecordingDocumentService(guard_state)
    module = _utils_module(store, document_service)

    document_ids = module._load_graph_write_source_ids(
        "tenant-1",
        "dataset-1",
    )

    assert store.search_offsets == [0, 1000]
    assert len(document_ids) == 1001
    assert document_ids[0] == "doc-0"
    assert "doc-last" in document_ids


@pytest.mark.asyncio
async def test_generate_subgraph_replaces_checkpoint_inside_one_document_guard():
    guard_state = {"active": False}
    store = _GuardedStore(guard_state)
    document_service = _RecordingDocumentService(guard_state)
    utils_module = _utils_module(store, document_service)
    module = _index_module(store, document_service, utils_module)

    class Extractor:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __call__(self, *_args, **_kwargs):
            return ([{"entity_name": "part", "description": "front link"}], [])

    result = await module.generate_subgraph(
        Extractor,
        "tenant-1",
        "dataset-1",
        "doc-1",
        ["content"],
        "Chinese",
        ["part"],
        object(),
        object(),
        lambda **_kwargs: None,
        task_id="task-1",
    )

    assert result.graph["source_id"] == ["doc-1"]
    assert document_service.calls == [(('doc-1',), "dataset-1")]
    assert [event[0] for event in store.events] == ["delete", "insert"]


def test_all_graphrag_doc_store_writes_are_inside_guarded_sync_callbacks():
    allowed_functions = {
        "_insert_chunks_sync",
        "insert_batch",
        "replace_graph_chunks",
        "replace_subgraph",
        "replace_community_reports",
    }
    observed_writes = []

    for path in (UTILS_PATH, INDEX_PATH):
        module = ast.parse(path.read_text(encoding="utf-8"))
        parents = {}
        for parent in ast.walk(module):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (
                isinstance(function, ast.Attribute)
                and function.attr in {"insert", "delete"}
                and isinstance(function.value, ast.Attribute)
                and function.value.attr == "docStoreConn"
            ):
                continue
            owner = parents.get(node)
            while owner is not None and not isinstance(
                owner,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                owner = parents.get(owner)
            observed_writes.append((path.name, node.lineno, owner.name if owner else None))

    assert observed_writes
    assert all(owner in allowed_functions for _, _, owner in observed_writes), observed_writes


@pytest.mark.asyncio
async def test_generate_subgraph_does_not_write_after_source_deletion():
    guard_state = {"active": False}
    store = _GuardedStore(guard_state)
    document_service = _RecordingDocumentService(guard_state, reject=True)
    utils_module = _utils_module(store, document_service)
    module = _index_module(store, document_service, utils_module)

    class Extractor:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __call__(self, *_args, **_kwargs):
            return ([{"entity_name": "part", "description": "front link"}], [])

    with pytest.raises(RuntimeError, match="canceled for deletion"):
        await module.generate_subgraph(
            Extractor,
            "tenant-1",
            "dataset-1",
            "doc-1",
            ["content"],
            "Chinese",
            ["part"],
            object(),
            object(),
            lambda **_kwargs: None,
            task_id="task-1",
        )

    assert document_service.calls == [(('doc-1',), "dataset-1")]
    assert store.events == []


@pytest.mark.asyncio
async def test_set_graph_replacement_uses_one_guard_for_exact_graph_source_union():
    guard_state = {"active": False}
    store = _GuardedStore(guard_state)
    document_service = _RecordingDocumentService(guard_state)
    module = _utils_module(store, document_service)
    graph = _MiniGraph()
    graph.graph["source_id"] = ["doc-b", "doc-a", "doc-b"]

    await module.set_graph(
        "tenant-1",
        "dataset-1",
        SimpleNamespace(llm_name="embedding-1"),
        graph,
        SimpleNamespace(
            added_updated_nodes=set(),
            added_updated_edges=set(),
            removed_nodes=set(),
            removed_edges=set(),
        ),
        lambda **_kwargs: None,
    )

    assert document_service.calls == [
        (("doc-a", "doc-b"), "dataset-1"),
    ]
    assert [event[0] for event in store.events] == [
        "delete",
        "insert",
        "insert",
    ]


@pytest.mark.asyncio
async def test_set_graph_does_not_replace_shared_graph_when_any_source_is_deleted():
    guard_state = {"active": False}
    store = _GuardedStore(guard_state)
    document_service = _RecordingDocumentService(guard_state, reject=True)
    module = _utils_module(store, document_service)
    graph = _MiniGraph()
    graph.graph["source_id"] = ["doc-live", "doc-deleting"]

    with pytest.raises(RuntimeError, match="canceled for deletion"):
        await module.set_graph(
            "tenant-1",
            "dataset-1",
            SimpleNamespace(llm_name="embedding-1"),
            graph,
            SimpleNamespace(
                added_updated_nodes=set(),
                added_updated_edges=set(),
                removed_nodes=set(),
                removed_edges=set(),
            ),
            lambda **_kwargs: None,
        )

    assert document_service.calls == [
        (("doc-deleting", "doc-live"), "dataset-1"),
    ]
    assert store.events == []


@pytest.mark.asyncio
async def test_set_graph_guards_sources_from_existing_and_replacement_snapshots():
    guard_state = {"active": False}
    store = _GuardedStore(
        guard_state,
        existing_fields={
            "old-graph": {
                "source_id": ["doc-old", "doc-shared"],
            }
        },
    )
    document_service = _RecordingDocumentService(guard_state)
    module = _utils_module(store, document_service)
    graph = _MiniGraph()
    graph.graph["source_id"] = ["doc-new", "doc-shared"]

    await module.set_graph(
        "tenant-1",
        "dataset-1",
        SimpleNamespace(llm_name="embedding-1"),
        graph,
        SimpleNamespace(
            added_updated_nodes=set(),
            added_updated_edges=set(),
            removed_nodes=set(),
            removed_edges=set(),
        ),
        lambda **_kwargs: None,
    )

    assert document_service.calls == [
        (("doc-new", "doc-old", "doc-shared"), "dataset-1"),
    ]


@pytest.mark.asyncio
async def test_community_report_replace_is_guarded_by_all_graph_sources():
    guard_state = {"active": False}
    store = _GuardedStore(guard_state)
    document_service = _RecordingDocumentService(guard_state)
    utils_module = _utils_module(store, document_service)
    module = _index_module(store, document_service, utils_module)

    class CommunityReportsExtractor:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(
                structured_output=[
                    {
                        "title": "Front stabilizer link",
                        "findings": [{"explanation": "source evidence"}],
                        "weight": 1,
                        "entities": ["D91"],
                    }
                ],
                output=["report"],
            )

    module.extract_community.__globals__["CommunityReportsExtractor"] = (
        CommunityReportsExtractor
    )
    graph = _MiniGraph()
    graph.graph["source_id"] = ["doc-c", "doc-a", "doc-b", "doc-a"]

    await module.extract_community(
        graph,
        "tenant-1",
        "dataset-1",
        None,
        object(),
        object(),
        lambda **_kwargs: None,
        task_id="task-1",
    )

    assert document_service.calls == [
        (("doc-a", "doc-b", "doc-c"), "dataset-1"),
    ]
    assert [event[0] for event in store.events] == ["insert"]


@pytest.mark.asyncio
async def test_community_replace_guards_existing_and_new_report_sources():
    guard_state = {"active": False}
    store = _GuardedStore(
        guard_state,
        existing_fields={
            "old-report": {
                "id": "old-report",
                "source_id": ["doc-old"],
            }
        },
    )
    document_service = _RecordingDocumentService(guard_state)
    utils_module = _utils_module(store, document_service)
    module = _index_module(store, document_service, utils_module)

    class CommunityReportsExtractor:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(
                structured_output=[
                    {
                        "title": "Current report",
                        "findings": [],
                        "weight": 1,
                        "entities": ["D91"],
                    }
                ],
                output=["report"],
            )

    module.extract_community.__globals__["CommunityReportsExtractor"] = (
        CommunityReportsExtractor
    )
    graph = _MiniGraph()
    graph.graph["source_id"] = ["doc-new"]

    await module.extract_community(
        graph,
        "tenant-1",
        "dataset-1",
        None,
        object(),
        object(),
        lambda **_kwargs: None,
        task_id="task-1",
    )

    assert document_service.calls == [
        (("doc-new", "doc-old"), "dataset-1"),
    ]
    assert [event[0] for event in store.events] == ["insert", "delete"]


@pytest.mark.asyncio
async def test_community_replace_reads_all_existing_report_source_pages():
    guard_state = {"active": False}
    first_page = {
        f"old-report-{index}": {"source_id": [f"doc-old-{index}"]}
        for index in range(1000)
    }
    store = _GuardedStore(
        guard_state,
        existing_pages={
            0: first_page,
            1000: {
                "last-old-report": {"source_id": ["doc-last-old"]},
            },
        },
    )
    document_service = _RecordingDocumentService(guard_state)
    utils_module = _utils_module(store, document_service)
    module = _index_module(store, document_service, utils_module)

    class CommunityReportsExtractor:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(
                structured_output=[],
                output=[],
            )

    module.extract_community.__globals__["CommunityReportsExtractor"] = (
        CommunityReportsExtractor
    )
    graph = _MiniGraph()
    graph.graph["source_id"] = ["doc-new"]

    await module.extract_community(
        graph,
        "tenant-1",
        "dataset-1",
        None,
        object(),
        object(),
        lambda **_kwargs: None,
        task_id="task-1",
    )

    assert store.search_offsets == [0, 1000]
    guarded_ids = document_service.calls[0][0]
    assert len(guarded_ids) == 1002
    assert "doc-last-old" in guarded_ids
    assert "doc-new" in guarded_ids


@pytest.mark.asyncio
async def test_community_replace_fails_closed_when_existing_sources_are_unknown():
    guard_state = {"active": False}
    store = _GuardedStore(
        guard_state,
        search_error=OSError("community search unavailable"),
    )
    document_service = _RecordingDocumentService(guard_state)
    utils_module = _utils_module(store, document_service)
    module = _index_module(store, document_service, utils_module)

    class CommunityReportsExtractor:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(
                structured_output=[],
                output=[],
            )

    module.extract_community.__globals__["CommunityReportsExtractor"] = (
        CommunityReportsExtractor
    )
    graph = _MiniGraph()
    graph.graph["source_id"] = ["doc-new"]

    with pytest.raises(OSError, match="community search unavailable"):
        await module.extract_community(
            graph,
            "tenant-1",
            "dataset-1",
            None,
            object(),
            object(),
            lambda **_kwargs: None,
            task_id="task-1",
        )

    assert document_service.calls == []
    assert store.events == []
