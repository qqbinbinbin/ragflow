import ast
from functools import wraps
from pathlib import Path


def _load_owned_connection_context():
    source_path = Path(__file__).parents[2] / "api" / "db" / "db_models.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "OwnedConnectionContext"
    )
    namespace = {"wraps": wraps}
    exec(
        compile(
            ast.Module(body=[class_node], type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    return namespace["OwnedConnectionContext"]


class _FakeDatabase:
    def __init__(self, *, closed):
        self.closed = closed
        self.connect_calls = 0
        self.close_calls = 0

    def is_closed(self):
        return self.closed

    def connect(self):
        self.connect_calls += 1
        self.closed = False

    def close(self):
        self.close_calls += 1
        self.closed = True


def test_nested_connection_context_does_not_close_outer_connection():
    OwnedConnectionContext = _load_owned_connection_context()
    database = _FakeDatabase(closed=True)

    with OwnedConnectionContext(database):
        with OwnedConnectionContext(database):
            assert database.connect_calls == 1
            assert database.close_calls == 0
        assert database.close_calls == 0

    assert database.close_calls == 1
    assert database.closed is True


def test_connection_context_does_not_close_connection_owned_by_caller():
    OwnedConnectionContext = _load_owned_connection_context()
    database = _FakeDatabase(closed=False)

    with OwnedConnectionContext(database):
        assert database.connect_calls == 0

    assert database.close_calls == 0
    assert database.closed is False


def test_connection_context_supports_decorator_usage():
    OwnedConnectionContext = _load_owned_connection_context()
    database = _FakeDatabase(closed=True)

    @OwnedConnectionContext(database)
    def read_value():
        assert database.closed is False
        return "ok"

    assert read_value() == "ok"
    assert database.connect_calls == 1
    assert database.close_calls == 1
