import ast
import logging
import re
from pathlib import Path

import pandas as pd
from dateutil.parser import parse as datetime_parse


def _load_column_data_type():
    source = Path("rag/app/table.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "datetime_parse": datetime_parse,
        "logging": logging,
        "pd": pd,
        "re": re,
    }
    exec(compile(module, "rag/app/table.py", "exec"), namespace)
    return namespace["column_data_type"]


def test_table_column_type_does_not_treat_three_digit_revision_as_datetime():
    column_data_type = _load_column_data_type()

    values, column_type = column_data_type([
        "100-01-09 00:00:00",
        "100-02-09 00:00:00",
        "100-03-09 00:00:00",
    ])

    assert column_type == "text"
    assert values == [
        "100-01-09 00:00:00",
        "100-02-09 00:00:00",
        "100-03-09 00:00:00",
    ]


def test_table_column_type_keeps_supported_four_digit_dates_as_datetime():
    column_data_type = _load_column_data_type()

    values, column_type = column_data_type([
        "2025-01-09 00:00:00",
        "2025-02-09 00:00:00",
        "2025-03-09 00:00:00",
    ])

    assert column_type == "datetime"
    assert values == [
        "2025-01-09 00:00:00",
        "2025-02-09 00:00:00",
        "2025-03-09 00:00:00",
    ]


def test_table_column_type_keeps_missing_scalars_out_of_content_and_type_vote():
    column_data_type = _load_column_data_type()

    values, column_type = column_data_type([
        float("nan"),
        pd.NA,
        "North",
    ])

    assert column_type == "text"
    assert values == [None, None, "North"]
