import base64
import binascii
import json
import struct
from json import JSONDecodeError
from typing import Any

import torch
from safetensors import SafetensorError
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors


_FORMAT_PREFIX = "res4lyf-conditioning-v2:"
_MAGIC = b"RES4LYF-CONDITIONING\x00"
_METADATA_LENGTH = struct.Struct(">I")
_MAX_ENCODED_SIZE = 512 * 1024 * 1024
_MAX_METADATA_SIZE = 8 * 1024 * 1024
_MAX_STRUCTURE_NODES = 100_000


def serialize_conditioning(conditioning: Any) -> str:
    """Serialize conditioning as JSON structure and non-executable tensor data."""
    tensors: dict[str, torch.Tensor] = {}
    structure = _encode_structure(conditioning, tensors)
    metadata = json.dumps(structure, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(metadata) > _MAX_METADATA_SIZE:
        raise ValueError("Conditioning metadata is too large")

    tensor_data = save_safetensors(tensors)
    payload = _MAGIC + _METADATA_LENGTH.pack(len(metadata)) + metadata + tensor_data
    encoded = base64.b64encode(payload).decode("ascii")
    if len(encoded) > _MAX_ENCODED_SIZE:
        raise ValueError("Conditioning payload is too large")
    return _FORMAT_PREFIX + encoded


def deserialize_conditioning(data: str) -> Any:
    """Load conditioning without invoking a pickle or Python object loader."""
    if not isinstance(data, str) or not data.startswith(_FORMAT_PREFIX):
        raise ValueError("Unsupported conditioning payload; legacy pickle payloads are not accepted")

    encoded = data[len(_FORMAT_PREFIX):]
    if len(encoded) > _MAX_ENCODED_SIZE:
        raise ValueError("Conditioning payload is too large")

    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Conditioning payload is not valid base64") from error

    header_size = len(_MAGIC) + _METADATA_LENGTH.size
    if len(payload) < header_size or not payload.startswith(_MAGIC):
        raise ValueError("Conditioning payload has an invalid format")

    metadata_size = _METADATA_LENGTH.unpack_from(payload, len(_MAGIC))[0]
    if metadata_size > _MAX_METADATA_SIZE or header_size + metadata_size > len(payload):
        raise ValueError("Conditioning payload has invalid metadata")

    metadata_end = header_size + metadata_size
    try:
        structure = json.loads(payload[header_size:metadata_end])
        tensors = load_safetensors(payload[metadata_end:])
    except (JSONDecodeError, SafetensorError, TypeError, ValueError) as error:
        raise ValueError("Conditioning payload is invalid") from error

    used_tensors: set[str] = set()
    try:
        conditioning = _decode_structure(structure, tensors, used_tensors, [_MAX_STRUCTURE_NODES])
    except (KeyError, RecursionError, TypeError, ValueError) as error:
        raise ValueError("Conditioning payload has an invalid structure") from error

    if used_tensors != set(tensors):
        raise ValueError("Conditioning payload contains unreferenced tensors")
    return conditioning


def _encode_structure(value: Any, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        key = str(len(tensors))
        tensors[key] = value.detach().cpu().contiguous()
        return {"type": "tensor", "key": key}

    if isinstance(value, list):
        return {"type": "list", "items": [_encode_structure(item, tensors) for item in value]}

    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode_structure(item, tensors) for item in value]}

    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Conditioning dictionaries must use string keys")
            items.append([key, _encode_structure(item, tensors)])
        return {"type": "dict", "items": items}

    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": "value", "value": value}

    raise TypeError(f"Conditioning contains unsupported type: {type(value).__name__}")


def _decode_structure(
    node: Any,
    tensors: dict[str, torch.Tensor],
    used_tensors: set[str],
    remaining_nodes: list[int],
) -> Any:
    remaining_nodes[0] -= 1
    if remaining_nodes[0] < 0:
        raise ValueError("Conditioning structure is too complex")
    if not isinstance(node, dict) or not isinstance(node.get("type"), str):
        raise ValueError("Conditioning structure node is invalid")

    node_type = node["type"]
    if node_type == "tensor":
        if set(node) != {"type", "key"} or not isinstance(node["key"], str):
            raise ValueError("Conditioning tensor reference is invalid")
        key = node["key"]
        tensor = tensors[key]
        used_tensors.add(key)
        return tensor

    if node_type in {"list", "tuple"}:
        if set(node) != {"type", "items"} or not isinstance(node["items"], list):
            raise ValueError("Conditioning sequence is invalid")
        items = [
            _decode_structure(item, tensors, used_tensors, remaining_nodes)
            for item in node["items"]
        ]
        return items if node_type == "list" else tuple(items)

    if node_type == "dict":
        if set(node) != {"type", "items"} or not isinstance(node["items"], list):
            raise ValueError("Conditioning dictionary is invalid")
        result = {}
        for item in node["items"]:
            if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
                raise ValueError("Conditioning dictionary item is invalid")
            if item[0] in result:
                raise ValueError("Conditioning dictionary contains duplicate keys")
            result[item[0]] = _decode_structure(item[1], tensors, used_tensors, remaining_nodes)
        return result

    if node_type == "value":
        if set(node) != {"type", "value"}:
            raise ValueError("Conditioning value is invalid")
        value = node["value"]
        if value is not None and not isinstance(value, (bool, int, float, str)):
            raise ValueError("Conditioning value type is invalid")
        return value

    raise ValueError(f"Unknown conditioning structure type: {node_type}")
