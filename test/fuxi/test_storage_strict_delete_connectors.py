import ast
import logging
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]

CONNECTORS = {
    "MINIO": ("rag/utils/minio_conn.py", "RAGFlowMinio"),
    "AWS_S3": ("rag/utils/s3_conn.py", "RAGFlowS3"),
    "OSS": ("rag/utils/oss_conn.py", "RAGFlowOSS"),
    "GCS": ("rag/utils/gcs_conn.py", "RAGFlowGCS"),
    "AZURE_SAS": ("rag/utils/azure_sas_conn.py", "RAGFlowAzureSasBlob"),
    "AZURE_SPN": ("rag/utils/azure_spn_conn.py", "RAGFlowAzureSpnBlob"),
    "OPENDAL": ("rag/utils/opendal_conn.py", "OpenDALStorage"),
}


class FakeNotFound(Exception):
    pass


class FakeClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3Error(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _identity_decorator(cls):
    return cls


def _load_class_module_light(relative_path, class_name):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[class_node], type_ignores=[])
    )
    namespace = {
        "AzureAuthorityHosts": SimpleNamespace(
            AZURE_PUBLIC_CLOUD="public",
            AZURE_CHINA="china",
            AZURE_GOVERNMENT="government",
            AZURE_GERMANY="germany",
        ),
        "BytesIO": BytesIO,
        "ClientError": FakeClientError,
        "CryptoUtil": object,
        "NotFound": FakeNotFound,
        "ResourceNotFoundError": FakeNotFound,
        "S3Error": FakeS3Error,
        "logging": logging,
        "singleton": _identity_decorator,
        "time": SimpleNamespace(sleep=lambda _seconds: None),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[class_name]


def _new_instance(relative_path, class_name, **attributes):
    cls = _load_class_module_light(relative_path, class_name)
    instance = object.__new__(cls)
    for name, value in attributes.items():
        setattr(instance, name, value)
    return instance


def _class_method_names(relative_path, class_name):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _class_method_node(relative_path, class_name, method_name):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def _load_method_light(relative_path, class_name, method_name, namespace=None):
    assert method_name in _class_method_names(relative_path, class_name), (
        f"{class_name}.{method_name} must be implemented"
    )
    method = deepcopy(
        _class_method_node(relative_path, class_name, method_name)
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    loaded = {} if namespace is None else dict(namespace)
    exec(
        compile(module, str(ROOT / relative_path), "exec"),
        loaded,
    )
    return loaded[method_name]


class _FakeQuery:
    def __init__(self, query_type, **kwargs):
        self.query_type = query_type
        self.kwargs = kwargs
        self.filter = []
        self.must = []
        self.must_not = []

    def to_dict(self):
        if self.query_type == "bool":
            return {
                "bool": {
                    "filter": [query.to_dict() for query in self.filter],
                    "must": [query.to_dict() for query in self.must],
                    "must_not": [query.to_dict() for query in self.must_not],
                }
            }
        return {self.query_type: self.kwargs}


class _FakeSearch:
    def query(self, query):
        self._query = query
        return self

    def to_dict(self):
        return {"query": self._query.to_dict()}


class _FakeInfinityException(Exception):
    def __init__(self, error_code):
        super().__init__(str(error_code))
        self.error_code = error_code


class _FakeResultRows:
    def __init__(self, rows):
        self._rows = list(rows)

    def fetchall(self):
        return list(self._rows)


class _FakeInfinityFrame:
    def __init__(self, empty):
        self.empty = empty


class _FakeInfinityTable:
    def __init__(self, *, deleted_rows=1, remaining=False):
        self.deleted_rows = deleted_rows
        self.remaining = remaining
        self.delete_error = None

    def delete(self, _filter):
        if self.delete_error is not None:
            raise self.delete_error
        return SimpleNamespace(deleted_rows=self.deleted_rows)

    def output(self, _fields):
        return self

    def filter(self, _filter):
        return self

    def to_df(self):
        return _FakeInfinityFrame(not self.remaining), None


class _FakeInfinityPool:
    def __init__(self, database):
        self.connection = SimpleNamespace(
            get_database=lambda _name: database,
        )
        self.released = []

    def get_conn(self):
        return self.connection

    def release_conn(self, connection):
        self.released.append(connection)


def test_docstore_strict_defaults_fail_closed():
    delete_strict = _load_method_light(
        "common/doc_store/doc_store_base.py",
        "DocStoreConnection",
        "delete_strict",
    )
    index_exist_strict = _load_method_light(
        "common/doc_store/doc_store_base.py",
        "DocStoreConnection",
        "index_exist_strict",
    )

    with pytest.raises(NotImplementedError):
        delete_strict(object(), {"doc_id": "doc-1"}, "index-1", "kb-1")
    with pytest.raises(NotImplementedError):
        index_exist_strict(object(), "index-1", "kb-1")


@pytest.mark.parametrize(
    ("relative_path", "class_name"),
    [
        ("rag/utils/opensearch_conn.py", "OSConnection"),
        ("rag/utils/infinity_conn.py", "InfinityConnection"),
        ("rag/utils/ob_conn.py", "OBConnection"),
    ],
    ids=["OpenSearch", "Infinity", "OceanBase-SeekDB"],
)
def test_docstore_strict_methods_do_not_fall_back_to_legacy_methods(
    relative_path,
    class_name,
):
    for method_name, forbidden_name in (
        ("delete_strict", "delete"),
        ("index_exist_strict", "index_exist"),
    ):
        method = _class_method_node(relative_path, class_name, method_name)
        assert not any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
            and call.func.attr == forbidden_name
            for call in ast.walk(method)
            if isinstance(call, ast.Call)
        )


def test_opensearch_strict_delete_propagates_and_confirms_readback():
    delete_strict = _load_method_light(
        "rag/utils/opensearch_conn.py",
        "OSConnection",
        "delete_strict",
        {
            "NotFoundError": FakeNotFound,
            "Q": _FakeQuery,
            "Search": _FakeSearch,
        },
    )
    index_exist_strict = _load_method_light(
        "rag/utils/opensearch_conn.py",
        "OSConnection",
        "index_exist_strict",
    )
    client = SimpleNamespace(
        delete_by_query=lambda **_kwargs: {
            "deleted": 1,
            "total": 1,
            "noops": 0,
            "timed_out": False,
            "failures": [],
            "version_conflicts": 0,
        },
        count=lambda **_kwargs: {"count": 0},
        indices=SimpleNamespace(exists=lambda **_kwargs: True),
    )
    store = SimpleNamespace(os=client)

    assert delete_strict(
        store, {"doc_id": "doc-1"}, "index-1", "kb-1"
    ) == 1
    assert index_exist_strict(store, "index-1", "kb-1") is True

    client.count = lambda **_kwargs: {"count": 1}
    with pytest.raises(RuntimeError, match="readback"):
        delete_strict(store, {"doc_id": "doc-1"}, "index-1", "kb-1")

    client.count = lambda **_kwargs: {"count": 0}
    client.delete_by_query = lambda **_kwargs: {
        "deleted": 1,
        "total": 2,
        "noops": 0,
        "timed_out": False,
        "failures": [],
        "version_conflicts": 0,
    }
    with pytest.raises(RuntimeError, match="incomplete"):
        delete_strict(store, {"doc_id": "doc-1"}, "index-1", "kb-1")

    client.delete_by_query = lambda **_kwargs: (_ for _ in ()).throw(
        FakeNotFound("missing index")
    )
    assert delete_strict(
        store, {"doc_id": "doc-1"}, "index-1", "kb-1"
    ) == 0

    marker = RuntimeError("opensearch unavailable")
    client.delete_by_query = lambda **_kwargs: (_ for _ in ()).throw(marker)
    with pytest.raises(RuntimeError, match="opensearch unavailable"):
        delete_strict(store, {"doc_id": "doc-1"}, "index-1", "kb-1")

    client.indices.exists = lambda **_kwargs: (_ for _ in ()).throw(marker)
    with pytest.raises(RuntimeError, match="opensearch unavailable"):
        index_exist_strict(store, "index-1", "kb-1")


def test_infinity_strict_delete_propagates_and_confirms_readback():
    namespace = {
        "ErrorCode": SimpleNamespace(TABLE_NOT_EXIST=3022),
        "InfinityException": _FakeInfinityException,
    }
    delete_strict = _load_method_light(
        "rag/utils/infinity_conn.py",
        "InfinityConnection",
        "delete_strict",
        namespace,
    )
    index_exist_strict = _load_method_light(
        "rag/utils/infinity_conn.py",
        "InfinityConnection",
        "index_exist_strict",
        namespace,
    )
    table = _FakeInfinityTable()
    database = SimpleNamespace(get_table=lambda _name: table)
    pool = _FakeInfinityPool(database)
    store = SimpleNamespace(
        connPool=pool,
        dbName="default_db",
        equivalent_condition_to_str=lambda _condition, _table: "doc_id = 'doc-1'",
    )

    assert delete_strict(
        store, {"doc_id": "doc-1"}, "index-1", "kb-1"
    ) == 1
    assert index_exist_strict(store, "index-1", "kb-1") is True

    table.remaining = True
    with pytest.raises(RuntimeError, match="readback"):
        delete_strict(store, {"doc_id": "doc-1"}, "index-1", "kb-1")

    table.remaining = False
    table.delete_error = RuntimeError("infinity unavailable")
    with pytest.raises(RuntimeError, match="infinity unavailable"):
        delete_strict(store, {"doc_id": "doc-1"}, "index-1", "kb-1")

    database.get_table = lambda _name: (_ for _ in ()).throw(
        _FakeInfinityException(3022)
    )
    assert delete_strict(
        store, {"doc_id": "doc-1"}, "index-1", "kb-1"
    ) == 0
    assert index_exist_strict(store, "index-1", "kb-1") is False

    database.get_table = lambda _name: (_ for _ in ()).throw(
        RuntimeError("infinity probe failed")
    )
    with pytest.raises(RuntimeError, match="infinity probe failed"):
        index_exist_strict(store, "index-1", "kb-1")


class _FakeOBClient:
    def __init__(self, rows):
        self.rows = list(rows)
        self.delete_error = None
        self.exists_error = None
        self.exists = True

    def check_table_exists(self, _table_name):
        if self.exists_error is not None:
            raise self.exists_error
        return self.exists

    def get(self, **_kwargs):
        return _FakeResultRows(self.rows)

    def delete(self, **_kwargs):
        if self.delete_error is not None:
            raise self.delete_error
        self.rows = []


@pytest.mark.parametrize("runtime_name", ["OceanBase", "SeekDB"])
def test_ob_family_strict_delete_propagates_and_confirms_readback(runtime_name):
    del runtime_name
    namespace = {"text": lambda value: value}
    delete_strict = _load_method_light(
        "rag/utils/ob_conn.py",
        "OBConnection",
        "delete_strict",
        namespace,
    )
    index_exist_strict = _load_method_light(
        "rag/utils/ob_conn.py",
        "OBConnection",
        "index_exist_strict",
    )
    client = _FakeOBClient([("chunk-1",)])
    store = SimpleNamespace(
        client=client,
        get_table_name=lambda index_name, _dataset_id: index_name,
        _get_dataset_id_field=lambda: "kb_id",
        _get_filters=lambda condition: [
            f"{key}={value}" for key, value in condition.items()
        ],
    )

    assert delete_strict(
        store, {"doc_id": "doc-1"}, "index-1", "kb-1"
    ) == 1
    assert index_exist_strict(store, "index-1", "kb-1") is True

    client.exists = False
    assert delete_strict(
        store, {"doc_id": "doc-1"}, "index-1", "kb-1"
    ) == 0
    assert index_exist_strict(store, "index-1", "kb-1") is False

    client.exists = True
    client.rows = [("chunk-1",)]
    client.delete = lambda **_kwargs: None
    with pytest.raises(RuntimeError, match="readback"):
        delete_strict(store, {"doc_id": "doc-1"}, "index-1", "kb-1")

    marker = RuntimeError("ob unavailable")
    client.delete = lambda **_kwargs: (_ for _ in ()).throw(marker)
    with pytest.raises(RuntimeError, match="ob unavailable"):
        delete_strict(store, {"doc_id": "doc-1"}, "index-1", "kb-1")

    client.exists_error = RuntimeError("ob probe failed")
    with pytest.raises(RuntimeError, match="ob probe failed"):
        index_exist_strict(store, "index-1", "kb-1")


def test_storage_factory_inventory_covers_every_runtime_backend_and_wrapper():
    source = (ROOT / "common/settings.py").read_text(encoding="utf-8")

    for storage_name, (_path, class_name) in CONNECTORS.items():
        assert f"Storage.{storage_name}: {class_name}" in source
    assert "create_encrypted_storage(storage_impl" in source


@pytest.mark.parametrize(
    ("relative_path", "class_name"),
    CONNECTORS.values(),
    ids=CONNECTORS.keys(),
)
def test_runtime_storage_backends_expose_strict_delete_contract(
    relative_path,
    class_name,
):
    method_names = _class_method_names(relative_path, class_name)

    assert {"rm_strict", "obj_exist_strict", "rm_prefix_strict"} <= method_names


@pytest.mark.parametrize(
    ("relative_path", "class_name"),
    CONNECTORS.values(),
    ids=CONNECTORS.keys(),
)
def test_strict_delete_methods_never_call_the_legacy_rm_method(
    relative_path,
    class_name,
):
    for method_name in ("rm_strict", "rm_prefix_strict"):
        method = _class_method_node(relative_path, class_name, method_name)
        assert not any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
            and call.func.attr == "rm"
            for call in ast.walk(method)
            if isinstance(call, ast.Call)
        )


class _FakeS3Client:
    def __init__(self, objects=()):
        self.objects = set(objects)
        self.delete_error = None
        self.delete_is_noop = False
        self.head_error = None
        self.list_error = None

    def delete_object(self, *, Bucket, Key):
        if self.delete_error is not None:
            raise self.delete_error
        if not self.delete_is_noop:
            self.objects.discard((Bucket, Key))

    def head_object(self, *, Bucket, Key):
        if self.head_error is not None:
            raise self.head_error
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("404")
        return {"ContentLength": 1}

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        client = self

        class Paginator:
            def paginate(self, *, Bucket, Prefix):
                if client.list_error is not None:
                    raise client.list_error
                keys = sorted(
                    key
                    for bucket, key in client.objects
                    if bucket == Bucket and key.startswith(Prefix)
                )
                return [{"Contents": [{"Key": key} for key in keys]}]

        return Paginator()


@pytest.mark.parametrize(
    ("relative_path", "class_name", "conn_factory"),
    [
        ("rag/utils/s3_conn.py", "RAGFlowS3", lambda client: [client]),
        ("rag/utils/oss_conn.py", "RAGFlowOSS", lambda client: client),
    ],
    ids=["AWS_S3", "OSS"],
)
def test_s3_compatible_strict_delete_verifies_absence_and_prefix_scope(
    relative_path,
    class_name,
    conn_factory,
):
    client = _FakeS3Client(
        {
            ("dataset", "scope/part-1"),
            ("dataset", "scope/part-2"),
            ("dataset", "other/keep"),
        }
    )
    storage = _new_instance(
        relative_path,
        class_name,
        conn=conn_factory(client),
        bucket=None,
        prefix_path=None,
    )

    assert storage.obj_exist_strict("dataset", "scope/part-1") is True
    storage.rm_strict("dataset", "scope/part-1")
    assert storage.obj_exist_strict("dataset", "scope/part-1") is False
    assert storage.rm_prefix_strict("dataset", "scope/") == 1
    assert client.objects == {("dataset", "other/keep")}

    client.objects.add(("dataset", "stale"))
    client.delete_is_noop = True
    with pytest.raises(OSError, match="incomplete"):
        storage.rm_strict("dataset", "stale")

    marker = RuntimeError("delete failed")
    client.delete_error = marker
    with pytest.raises(RuntimeError, match="delete failed"):
        storage.rm_strict("dataset", "stale")

    client.delete_error = None
    client.objects.add(("dataset", "scope/stale"))
    with pytest.raises(OSError, match="prefix deletion was incomplete"):
        storage.rm_prefix_strict("dataset", "scope/")

    client.delete_is_noop = False
    client.delete_error = marker
    with pytest.raises(RuntimeError, match="delete failed"):
        storage.rm_prefix_strict("dataset", "scope/")

    client.delete_error = None
    client.head_error = RuntimeError("head failed")
    with pytest.raises(RuntimeError, match="head failed"):
        storage.obj_exist_strict("dataset", "stale")

    client.head_error = None
    client.list_error = RuntimeError("list failed")
    with pytest.raises(RuntimeError, match="list failed"):
        storage.rm_prefix_strict("dataset", "scope/")


def test_s3_strict_prefix_delete_preserves_logical_bucket_in_default_bucket_mode():
    client = _FakeS3Client(
        {
            ("physical", "root/dataset-a/scope/part-1"),
            ("physical", "root/dataset-b/scope/keep"),
        }
    )
    storage = _new_instance(
        "rag/utils/s3_conn.py",
        "RAGFlowS3",
        conn=[client],
        bucket="physical",
        prefix_path="root",
    )

    assert storage.rm_prefix_strict("dataset-a", "scope/") == 1
    assert client.objects == {("physical", "root/dataset-b/scope/keep")}


class _FakeMinioClient:
    def __init__(self, objects=()):
        self.objects = set(objects)
        self.delete_error = None
        self.delete_is_noop = False
        self.list_error = None
        self.stat_error = None

    def bucket_exists(self, bucket):
        return any(stored_bucket == bucket for stored_bucket, _name in self.objects)

    def stat_object(self, bucket, name):
        if self.stat_error is not None:
            raise self.stat_error
        if (bucket, name) not in self.objects:
            raise FakeS3Error("NoSuchKey")
        return SimpleNamespace(object_name=name)

    def remove_object(self, bucket, name):
        if self.delete_error is not None:
            raise self.delete_error
        if not self.delete_is_noop:
            self.objects.discard((bucket, name))

    def list_objects(self, bucket, *, prefix, recursive):
        assert recursive is True
        if self.list_error is not None:
            raise self.list_error
        return [
            SimpleNamespace(object_name=name)
            for stored_bucket, name in sorted(self.objects)
            if stored_bucket == bucket and name.startswith(prefix)
        ]


def test_minio_strict_delete_verifies_absence_and_propagates_errors():
    client = _FakeMinioClient(
        {
            ("dataset", "scope/part-1"),
            ("dataset", "scope/part-2"),
            ("dataset", "other/keep"),
        }
    )
    storage = _new_instance(
        "rag/utils/minio_conn.py",
        "RAGFlowMinio",
        conn=client,
        bucket=None,
        prefix_path=None,
    )

    assert storage.obj_exist_strict("dataset", "scope/part-1") is True
    storage.rm_strict("dataset", "scope/part-1")
    assert storage.rm_prefix_strict("dataset", "scope/") == 1
    assert client.objects == {("dataset", "other/keep")}

    client.objects.add(("dataset", "stale"))
    client.delete_is_noop = True
    with pytest.raises(OSError, match="incomplete"):
        storage.rm_strict("dataset", "stale")

    client.delete_error = RuntimeError("delete failed")
    with pytest.raises(RuntimeError, match="delete failed"):
        storage.rm_strict("dataset", "stale")

    client.delete_error = None
    client.objects.add(("dataset", "scope/stale"))
    with pytest.raises(OSError, match="prefix deletion was incomplete"):
        storage.rm_prefix_strict("dataset", "scope/")

    client.delete_is_noop = False
    client.delete_error = RuntimeError("delete failed")
    with pytest.raises(RuntimeError, match="delete failed"):
        storage.rm_prefix_strict("dataset", "scope/")

    client.delete_error = None
    client.stat_error = RuntimeError("stat failed")
    with pytest.raises(RuntimeError, match="stat failed"):
        storage.obj_exist_strict("dataset", "stale")

    client.stat_error = None
    client.list_error = RuntimeError("list failed")
    with pytest.raises(RuntimeError, match="list failed"):
        storage.rm_prefix_strict("dataset", "scope/")


def test_minio_strict_prefix_delete_preserves_logical_bucket_in_default_bucket_mode():
    client = _FakeMinioClient(
        {
            ("physical", "root/dataset-a/scope/part-1"),
            ("physical", "root/dataset-b/scope/keep"),
        }
    )
    storage = _new_instance(
        "rag/utils/minio_conn.py",
        "RAGFlowMinio",
        conn=client,
        bucket="physical",
        prefix_path="root",
    )

    assert storage.rm_prefix_strict("dataset-a", "scope/") == 1
    assert client.objects == {("physical", "root/dataset-b/scope/keep")}


class _ObjectSetBackend:
    def __init__(self, objects=()):
        self.objects = set(objects)
        self.delete_error = None
        self.delete_is_noop = False
        self.exists_error = None

    def delete(self, name):
        if self.delete_error is not None:
            raise self.delete_error
        if not self.delete_is_noop:
            self.objects.discard(name)

    def exists(self, name):
        if self.exists_error is not None:
            raise self.exists_error
        return name in self.objects


class _FakeGCSBlob:
    def __init__(self, backend, name):
        self.backend = backend
        self.name = name

    def delete(self):
        self.backend.delete(self.name)

    def exists(self):
        return self.backend.exists(self.name)


class _FakeGCSBucket:
    def __init__(self, backend):
        self.backend = backend

    def blob(self, name):
        return _FakeGCSBlob(self.backend, name)


class _FakeGCSClient:
    def __init__(self, backend):
        self.backend = backend
        self.list_error = None

    def bucket(self, _bucket_name):
        return _FakeGCSBucket(self.backend)

    def list_blobs(self, _bucket_name, *, prefix):
        if self.list_error is not None:
            raise self.list_error
        return [
            _FakeGCSBlob(self.backend, name)
            for name in sorted(self.backend.objects)
            if name.startswith(prefix)
        ]


def test_gcs_strict_delete_verifies_absence_and_prefix_scope():
    backend = _ObjectSetBackend(
        {"dataset/scope/part-1", "dataset/scope/part-2", "dataset/other/keep"}
    )
    storage = _new_instance(
        "rag/utils/gcs_conn.py",
        "RAGFlowGCS",
        client=_FakeGCSClient(backend),
        bucket_name="physical",
    )

    assert storage.obj_exist_strict("dataset", "scope/part-1") is True
    storage.rm_strict("dataset", "scope/part-1")
    assert storage.rm_prefix_strict("dataset", "scope/") == 1
    assert backend.objects == {"dataset/other/keep"}

    backend.objects.add("dataset/stale")
    backend.delete_is_noop = True
    with pytest.raises(OSError, match="incomplete"):
        storage.rm_strict("dataset", "stale")

    backend.delete_is_noop = False
    backend.delete_error = RuntimeError("delete failed")
    with pytest.raises(RuntimeError, match="delete failed"):
        storage.rm_strict("dataset", "stale")

    backend.delete_error = None
    backend.delete_is_noop = True
    backend.objects.add("dataset/scope/stale")
    with pytest.raises(OSError, match="prefix deletion was incomplete"):
        storage.rm_prefix_strict("dataset", "scope/")

    backend.delete_is_noop = False
    backend.delete_error = RuntimeError("delete failed")
    with pytest.raises(RuntimeError, match="delete failed"):
        storage.rm_prefix_strict("dataset", "scope/")

    backend.delete_error = None
    backend.exists_error = RuntimeError("exists failed")
    with pytest.raises(RuntimeError, match="exists failed"):
        storage.obj_exist_strict("dataset", "stale")

    backend.exists_error = None
    storage.client.list_error = RuntimeError("list failed")
    with pytest.raises(RuntimeError, match="list failed"):
        storage.rm_prefix_strict("dataset", "scope/")


class _FakeAzureBlobClient:
    def __init__(self, backend, name):
        self.backend = backend
        self.name = name

    def exists(self):
        return self.backend.exists(self.name)


class _FakeAzureContainer:
    def __init__(self, backend):
        self.backend = backend
        self.list_error = None

    def delete_blob(self, name):
        self.backend.delete(name)

    def get_blob_client(self, name):
        return _FakeAzureBlobClient(self.backend, name)

    def list_blobs(self, *, name_starts_with):
        if self.list_error is not None:
            raise self.list_error
        return [
            SimpleNamespace(name=name)
            for name in sorted(self.backend.objects)
            if name.startswith(name_starts_with)
        ]


def test_azure_sas_strict_delete_verifies_absence_and_prefix_scope():
    backend = _ObjectSetBackend(
        {"dataset/scope/part-1", "dataset/scope/part-2", "dataset/other/keep"}
    )
    storage = _new_instance(
        "rag/utils/azure_sas_conn.py",
        "RAGFlowAzureSasBlob",
        conn=_FakeAzureContainer(backend),
    )

    assert storage.obj_exist_strict("dataset", "scope/part-1") is True
    storage.rm_strict("dataset", "scope/part-1")
    assert storage.rm_prefix_strict("dataset", "scope/") == 1
    assert backend.objects == {"dataset/other/keep"}

    backend.objects.add("dataset/stale")
    backend.delete_is_noop = True
    with pytest.raises(OSError, match="incomplete"):
        storage.rm_strict("dataset", "stale")

    backend.delete_is_noop = False
    backend.delete_error = RuntimeError("delete failed")
    with pytest.raises(RuntimeError, match="delete failed"):
        storage.rm_strict("dataset", "stale")

    backend.delete_error = None
    backend.delete_is_noop = True
    backend.objects.add("dataset/scope/stale")
    with pytest.raises(OSError, match="prefix deletion was incomplete"):
        storage.rm_prefix_strict("dataset", "scope/")

    backend.delete_is_noop = False
    backend.delete_error = RuntimeError("delete failed")
    with pytest.raises(RuntimeError, match="delete failed"):
        storage.rm_prefix_strict("dataset", "scope/")

    backend.delete_error = None
    backend.exists_error = RuntimeError("exists failed")
    with pytest.raises(RuntimeError, match="exists failed"):
        storage.obj_exist_strict("dataset", "stale")

    backend.exists_error = None
    storage.conn.list_error = RuntimeError("list failed")
    with pytest.raises(RuntimeError, match="list failed"):
        storage.rm_prefix_strict("dataset", "scope/")


class _FakeAzureFileSystem:
    def __init__(self, backend):
        self.backend = backend
        self.list_error = None

    def delete_file(self, name):
        self.backend.delete(name)

    def get_file_client(self, name):
        return _FakeAzureBlobClient(self.backend, name)

    def get_paths(self, *, path, recursive):
        assert recursive is True
        if self.list_error is not None:
            raise self.list_error
        return [
            SimpleNamespace(name=name, is_directory=False)
            for name in sorted(self.backend.objects)
            if name.startswith(path)
        ]


def test_azure_spn_strict_delete_verifies_absence_and_prefix_scope():
    backend = _ObjectSetBackend(
        {"dataset/scope/part-1", "dataset/scope/part-2", "dataset/other/keep"}
    )
    storage = _new_instance(
        "rag/utils/azure_spn_conn.py",
        "RAGFlowAzureSpnBlob",
        conn=_FakeAzureFileSystem(backend),
    )

    assert storage.obj_exist_strict("dataset", "scope/part-1") is True
    storage.rm_strict("dataset", "scope/part-1")
    assert storage.rm_prefix_strict("dataset", "scope/") == 1
    assert backend.objects == {"dataset/other/keep"}

    backend.objects.add("dataset/stale")
    backend.delete_is_noop = True
    with pytest.raises(OSError, match="incomplete"):
        storage.rm_strict("dataset", "stale")

    backend.delete_is_noop = False
    backend.delete_error = RuntimeError("delete failed")
    with pytest.raises(RuntimeError, match="delete failed"):
        storage.rm_strict("dataset", "stale")

    backend.delete_error = None
    backend.delete_is_noop = True
    backend.objects.add("dataset/scope/stale")
    with pytest.raises(OSError, match="prefix deletion was incomplete"):
        storage.rm_prefix_strict("dataset", "scope/")

    backend.delete_is_noop = False
    backend.delete_error = RuntimeError("delete failed")
    with pytest.raises(RuntimeError, match="delete failed"):
        storage.rm_prefix_strict("dataset", "scope/")

    backend.delete_error = None
    backend.exists_error = RuntimeError("exists failed")
    with pytest.raises(RuntimeError, match="exists failed"):
        storage.obj_exist_strict("dataset", "stale")


def test_azure_spn_missing_prefix_is_an_idempotent_empty_delete():
    backend = _ObjectSetBackend()
    conn = _FakeAzureFileSystem(backend)
    conn.list_error = FakeNotFound("missing directory")
    storage = _new_instance(
        "rag/utils/azure_spn_conn.py",
        "RAGFlowAzureSpnBlob",
        conn=conn,
    )

    assert storage.rm_prefix_strict("dataset", "scope/") == 0

    conn.list_error = RuntimeError("list failed")
    with pytest.raises(RuntimeError, match="list failed"):
        storage.rm_prefix_strict("dataset", "scope/")


class _FakeOpenDALOperator:
    def __init__(self, backend):
        self.backend = backend
        self.scan_error = None

    def delete(self, name):
        self.backend.delete(name)

    def exists(self, name):
        return self.backend.exists(name)

    def scan(self, prefix):
        if self.scan_error is not None:
            raise self.scan_error
        return [
            name
            for name in sorted(self.backend.objects)
            if name.startswith(prefix)
        ]


class _FakeRelativeOpenDALOperator(_FakeOpenDALOperator):
    def scan(self, prefix):
        return [
            SimpleNamespace(path=lambda name=name: name.removeprefix(prefix))
            for name in sorted(self.backend.objects)
            if name.startswith(prefix)
        ]


def test_opendal_strict_delete_verifies_absence_and_prefix_scope():
    backend = _ObjectSetBackend(
        {"dataset/scope/part-1", "dataset/scope/part-2", "dataset/other/keep"}
    )
    storage = _new_instance(
        "rag/utils/opendal_conn.py",
        "OpenDALStorage",
        _operator=_FakeOpenDALOperator(backend),
    )

    assert storage.obj_exist_strict("dataset", "scope/part-1") is True
    storage.rm_strict("dataset", "scope/part-1")
    assert storage.rm_prefix_strict("dataset", "scope/") == 1
    assert backend.objects == {"dataset/other/keep"}

    backend.objects.add("dataset/stale")
    backend.delete_is_noop = True
    with pytest.raises(OSError, match="incomplete"):
        storage.rm_strict("dataset", "stale")

    backend.delete_is_noop = False
    backend.delete_error = RuntimeError("delete failed")
    with pytest.raises(RuntimeError, match="delete failed"):
        storage.rm_strict("dataset", "stale")

    backend.delete_error = None
    backend.delete_is_noop = True
    backend.objects.add("dataset/scope/stale")
    with pytest.raises(OSError, match="prefix deletion was incomplete"):
        storage.rm_prefix_strict("dataset", "scope/")

    backend.delete_is_noop = False
    backend.delete_error = RuntimeError("delete failed")
    with pytest.raises(RuntimeError, match="delete failed"):
        storage.rm_prefix_strict("dataset", "scope/")

    backend.delete_error = None
    backend.exists_error = RuntimeError("exists failed")
    with pytest.raises(RuntimeError, match="exists failed"):
        storage.obj_exist_strict("dataset", "stale")

    backend.exists_error = None
    storage._operator.scan_error = RuntimeError("scan failed")
    with pytest.raises(RuntimeError, match="scan failed"):
        storage.rm_prefix_strict("dataset", "scope/")


def test_opendal_prefix_delete_accepts_relative_entry_paths():
    backend = _ObjectSetBackend(
        {"dataset/scope/part-1", "dataset/other/keep"}
    )
    storage = _new_instance(
        "rag/utils/opendal_conn.py",
        "OpenDALStorage",
        _operator=_FakeRelativeOpenDALOperator(backend),
    )

    assert storage.rm_prefix_strict("dataset", "scope/") == 1
    assert backend.objects == {"dataset/other/keep"}


def test_encrypted_wrapper_requires_and_delegates_strict_backend_methods():
    wrapper = _new_instance(
        "rag/utils/encrypted_storage.py",
        "EncryptedStorageWrapper",
        storage_impl=SimpleNamespace(rm=lambda *_args: pytest.fail("legacy rm used")),
    )

    with pytest.raises(RuntimeError, match="strict object deletion"):
        wrapper.rm_strict("dataset", "object", "tenant")
    with pytest.raises(RuntimeError, match="strict prefix deletion"):
        wrapper.rm_prefix_strict("dataset", "scope/", "tenant")
    with pytest.raises(RuntimeError, match="strict object existence"):
        wrapper.obj_exist_strict("dataset", "object", "tenant")

    calls = []
    wrapper.storage_impl = SimpleNamespace(
        rm_strict=lambda *args: calls.append(("rm", args)),
        rm_prefix_strict=lambda *args: calls.append(("prefix", args)) or 2,
        obj_exist_strict=lambda *args: calls.append(("exists", args)) or False,
    )
    wrapper.rm_strict("dataset", "object", "tenant")
    assert wrapper.rm_prefix_strict("dataset", "scope/", "tenant") == 2
    assert wrapper.obj_exist_strict("dataset", "object", "tenant") is False
    assert calls == [
        ("rm", ("dataset", "object", "tenant")),
        ("prefix", ("dataset", "scope/", "tenant")),
        ("exists", ("dataset", "object", "tenant")),
    ]
