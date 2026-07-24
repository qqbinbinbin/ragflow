import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read_first(*relative_paths: str) -> str:
    for relative_path in relative_paths:
        path = ROOT / relative_path
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise AssertionError(f"none of the maintained paths exist: {relative_paths}")


def test_document_downloads_fail_explicitly_when_storage_object_is_missing():
    document_source = _read_first("api/apps/restful_apis/document_api.py")
    attachment_source = _read_first("api/apps/restful_apis/agent_api.py")

    assert "Document object is missing from storage" in document_source
    assert "Attachment object is missing from storage" in attachment_source


def test_queue_tasks_terminalizes_missing_source_and_empty_task_plan():
    source = (ROOT / "api/db/services/task_service.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    queue_tasks = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "queue_tasks"
    )
    function_source = ast.get_source_segment(source, queue_tasks) or ""

    assert "Failed to load PDF from object storage before queueing tasks" in function_source
    assert "Failed to load spreadsheet from object storage before queueing tasks" in function_source
    assert "No parsing tasks were generated for this document" in function_source
    assert function_source.count("TaskStatus.FAIL.value") >= 3
    assert function_source.count('"progress": -1') >= 3


def test_entrypoint_ensures_configured_minio_bucket_before_servers_start():
    source = (ROOT / "docker/entrypoint.sh").read_text(encoding="utf-8")

    definition = source.index("function ensure_minio_bucket()")
    invocation = source.index("ensure_minio_bucket", definition + 1)
    webserver = source.index('if [[ "${ENABLE_WEBSERVER}" -eq 1 ]]')
    assert definition < invocation < webserver
    assert "bucket_exists" in source
    assert "make_bucket" in source
