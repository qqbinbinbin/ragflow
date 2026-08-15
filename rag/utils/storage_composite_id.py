#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


def parse_storage_composite_id(composite_id: str) -> tuple[str, str] | None:
    """Split a ``{bucket}-{object_key}`` storage ID on the first hyphen."""

    parts = composite_id.split("-", 1)
    if len(parts) != 2 or not parts[0] or not parts[1] or composite_id.endswith("-"):
        return None
    return parts[0], parts[1]
