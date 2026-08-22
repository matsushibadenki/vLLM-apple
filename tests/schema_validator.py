from __future__ import annotations

from datetime import datetime
from typing import Any


SUPPORTED_KEYWORDS = {
    "$id",
    "$schema",
    "additionalProperties",
    "const",
    "description",
    "enum",
    "format",
    "items",
    "maximum",
    "minimum",
    "minLength",
    "properties",
    "required",
    "title",
    "type",
}


class SchemaValidationError(AssertionError):
    pass


def ensure_supported_schema(schema: dict[str, Any], path: str = "$schema") -> None:
    unsupported = set(schema) - SUPPORTED_KEYWORDS
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"{path}: unsupported JSON Schema keyword(s): {names}")
    if schema.get("format") not in {None, "date-time"}:
        raise ValueError(f"{path}: unsupported JSON Schema format {schema['format']!r}")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError(f"{path}.properties must be an object")
    for name, child in properties.items():
        if not isinstance(child, dict):
            raise ValueError(f"{path}.properties.{name} must be an object")
        ensure_supported_schema(child, f"{path}.properties.{name}")
    items = schema.get("items")
    if items is not None:
        if not isinstance(items, dict):
            raise ValueError(f"{path}.items must be an object")
        ensure_supported_schema(items, f"{path}.items")
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        ensure_supported_schema(additional, f"{path}.additionalProperties")


def validate_instance(instance: Any, schema: dict[str, Any], path: str = "$") -> None:
    ensure_supported_schema(schema)
    _validate(instance, schema, path)


def _validate(instance: Any, schema: dict[str, Any], path: str) -> None:
    if "const" in schema and not _json_equal(instance, schema["const"]):
        raise SchemaValidationError(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and not any(_json_equal(instance, value) for value in schema["enum"]):
        raise SchemaValidationError(f"{path}: value {instance!r} is not in enum")

    expected_types = schema.get("type")
    if expected_types is not None:
        names = [expected_types] if isinstance(expected_types, str) else expected_types
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"{path}: type must be a string or string array")
        if not any(_matches_type(instance, name) for name in names):
            raise SchemaValidationError(
                f"{path}: expected {' or '.join(names)}, got {type(instance).__name__}"
            )

    if isinstance(instance, dict):
        required = schema.get("required", [])
        missing = [name for name in required if name not in instance]
        if missing:
            raise SchemaValidationError(f"{path}: missing required properties {missing}")
        properties = schema.get("properties", {})
        for name, child_schema in properties.items():
            if name in instance:
                _validate(instance[name], child_schema, f"{path}.{name}")
        additional = schema.get("additionalProperties", True)
        unknown = set(instance) - set(properties)
        if additional is False and unknown:
            raise SchemaValidationError(
                f"{path}: additional properties are not allowed: {sorted(unknown)}"
            )
        if isinstance(additional, dict):
            for name in unknown:
                _validate(instance[name], additional, f"{path}.{name}")

    if isinstance(instance, list) and isinstance(schema.get("items"), dict):
        for index, value in enumerate(instance):
            _validate(value, schema["items"], f"{path}[{index}]")

    if isinstance(instance, str):
        minimum_length = schema.get("minLength")
        if minimum_length is not None and len(instance) < minimum_length:
            raise SchemaValidationError(f"{path}: string is shorter than {minimum_length}")
        if schema.get("format") == "date-time":
            try:
                parsed = datetime.fromisoformat(instance.replace("Z", "+00:00"))
            except ValueError as error:
                raise SchemaValidationError(f"{path}: invalid RFC 3339 date-time") from error
            if parsed.tzinfo is None:
                raise SchemaValidationError(f"{path}: date-time must include a timezone")

    if _matches_type(instance, "number"):
        if "minimum" in schema and instance < schema["minimum"]:
            raise SchemaValidationError(f"{path}: value is below minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            raise SchemaValidationError(f"{path}: value is above maximum")


def _matches_type(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    raise ValueError(f"unsupported JSON Schema type: {name}")


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right
