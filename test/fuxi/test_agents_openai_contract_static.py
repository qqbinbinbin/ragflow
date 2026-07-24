#!/usr/bin/env python3
"""Static contract check for FUXI Lane C RAGFlow Agent OpenAI route."""

from __future__ import annotations

import ast
from pathlib import Path


BACKWARD_COMPAT_PATH = Path(__file__).resolve().parents[2] / "api/apps/backward_compat.py"
AGENT_API_PATH = Path(__file__).resolve().parents[2] / "api/apps/restful_apis/agent_api.py"
LLM_PATH = Path(__file__).resolve().parents[2] / "agent/component/llm.py"
RETRIEVAL_PATH = Path(__file__).resolve().parents[2] / "agent/tools/retrieval.py"
CANVAS_PATH = Path(__file__).resolve().parents[2] / "agent/canvas.py"
ROUTE = "/agents_openai/<agent_id>/chat/completions"
LEGACY_HANDLER = "deprecated_agents_openai_chat_completions"
HANDLER = "agent_chat_completion"


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def main() -> int:
    legacy_source = BACKWARD_COMPAT_PATH.read_text(encoding="utf-8")
    legacy_tree = ast.parse(legacy_source)
    legacy_handlers = {
        node.name: node
        for node in ast.walk(legacy_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    legacy_handler = legacy_handlers.get(LEGACY_HANDLER)
    assert legacy_handler is not None, f"missing handler {LEGACY_HANDLER}"
    legacy_routes = [
        decorator.args[0].value
        for decorator in legacy_handler.decorator_list
        if isinstance(decorator, ast.Call)
        and getattr(decorator.func, "attr", "") == "route"
        and decorator.args
        and isinstance(decorator.args[0], ast.Constant)
    ]
    assert ROUTE in legacy_routes, f"missing route {ROUTE}"
    assert "agent_api.agent_chat_completion" in legacy_source
    assert 'req["openai-compatible"] = True' in legacy_source

    source = AGENT_API_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    handlers = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    handler = handlers.get(HANDLER)
    assert handler is not None, f"missing handler {HANDLER}"

    route_decorators = [
        decorator
        for decorator in handler.decorator_list
        if isinstance(decorator, ast.Call)
        and getattr(decorator.func, "attr", "") == "route"
    ]
    routes = [decorator.args[0].value for decorator in route_decorators if decorator.args and isinstance(decorator.args[0], ast.Constant)]
    assert "/agents/chat/completions" in routes

    decorator_names = {_decorator_name(decorator) for decorator in handler.decorator_list}
    assert "login_required" in decorator_names, "agent route must use maintained auth"
    assert "add_tenant_id_to_kwargs" in decorator_names, "agent route must inject authorized tenant scope"

    calls = {
        node.func.id
        for node in ast.walk(handler)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "completion_openai" in calls, "agents_openai route must reuse completion_openai"
    assert "_build_sse_response" in calls, "streaming agent route must return an SSE response"

    imported_names = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.asname or alias.name for alias in node.names)
    assert "completion_openai" in imported_names, "session.py must import completion_openai"

    arg_names = [arg.arg for arg in handler.args.args]
    assert arg_names[:2] == ["tenant_id", "agent_id"], "handler must receive tenant_id and agent_id"

    canvas_service_path = Path(__file__).resolve().parents[2] / "api/db/services/canvas_service.py"
    canvas_service_source = canvas_service_path.read_text(encoding="utf-8")
    assert (
        '"workflow_finished"' in canvas_service_source
        and "fallback_content" in canvas_service_source
    ), "OpenAI agent streaming must emit final content when DSL has no Message node"

    llm_source = LLM_PATH.read_text(encoding="utf-8")
    llm_tree = ast.parse(llm_source)
    class_names = {
        node.name
        for node in ast.walk(llm_tree)
        if isinstance(node, ast.ClassDef)
    }
    assert "GenerateParam" in class_names, "FUXI legacy DSL needs GenerateParam"
    assert "Generate" in class_names, "FUXI legacy DSL needs Generate"
    assert (
        "{input}" in llm_source and "formalized_content" in llm_source
    ), "Generate must map legacy {input} to upstream retrieval output"
    assert (
        "{begin@query}" in llm_source and "{sys.query}" in llm_source
    ), "Generate must map legacy {begin@query} to sys.query"

    retrieval_source = RETRIEVAL_PATH.read_text(encoding="utf-8")
    assert (
        '"default": "{sys.query}"' in retrieval_source
    ), "Retrieval without explicit query must default to sys.query"

    canvas_source = CANVAS_PATH.read_text(encoding="utf-8")
    assert (
        'setdefault("retrieval"' in canvas_source
        and 'setdefault("globals"' in canvas_source
    ), "Canvas must hydrate legacy FUXI DSL defaults before load"

    print("agents_openai_contract_static=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
