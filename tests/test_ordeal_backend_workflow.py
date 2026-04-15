"""Ordeal chaos tests for backend-orchestrator workflow semantics.

This models the interaction boundary between retrain-as-orchestrator and a
remote execution backend:

- train steps advance the backend policy version
- checkpoint makes the current policy visible to sampling
- sample sees only checkpointed weights, not uncheckpointed training updates
- save/load roundtrips restore backend-visible state
- ambiguous post-commit failures on train retry safely via request-id dedupe

The model is intentionally scalar-safe: no GTPO/SEPA assumptions are required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import hypothesis.strategies as st
from hypothesis import settings
from ordeal import ChaosTest, always, invariant, reachable, rule, sometimes
from ordeal.faults import LambdaFault
from ordeal.faults import io as io_faults
from ordeal.faults import network, timing


class BackendProtocolError(RuntimeError):
    """Raised when the simulated transport returns malformed payloads."""


_DROP_TRAIN_ACK = False
_CHECKPOINT_EMPTY_PAYLOAD = False
_CHECKPOINT_MISSING_VISIBLE_VERSION = False
_SAMPLE_INVALID_JSON = False
_SAMPLE_MISSING_GROUPS = False
_SAMPLE_WRONG_GROUP_COUNT = False
_SAMPLE_GROUP_NOT_LIST = False
_SAMPLE_WRONG_COMPLETION_COUNT = False
_SAMPLE_ENTRY_NOT_OBJECT = False
_SAMPLE_MISSING_TOKEN_FIELDS = False
_SAMPLE_ENTRY_VERSION_MISMATCH = False
_SAMPLE_BAD_LOGPROB_LENGTH = False
_TRAIN_MISSING_DEDUP = False
_SAVE_NON_OBJECT_PAYLOAD = False
_SAVE_MISSING_REF = False
_LOAD_MISSING_CURRENT_VERSION = False
_READBACK_BAD_CURRENT_VERSION_TYPE = False
_READBACK_BAD_VISIBLE_VERSION_TYPE = False
_READBACK_VISIBLE_AHEAD = False
_READBACK_BAD_APPLIED_REQUESTS_TYPE = False
_READBACK_NONCONTIGUOUS_REQUESTS = False
_READBACK_MISMATCHED_CURRENT_VERSION = False
_READBACK_BAD_REQUEST_ID_TYPE = False
_READBACK_BAD_REQUEST_VERSION_TYPE = False
_READBACK_BAD_CHECKPOINT_NAME_TYPE = False

_PROTOCOL_FLAG_NAMES = (
    "_CHECKPOINT_EMPTY_PAYLOAD",
    "_CHECKPOINT_MISSING_VISIBLE_VERSION",
    "_SAMPLE_INVALID_JSON",
    "_SAMPLE_MISSING_GROUPS",
    "_SAMPLE_WRONG_GROUP_COUNT",
    "_SAMPLE_GROUP_NOT_LIST",
    "_SAMPLE_WRONG_COMPLETION_COUNT",
    "_SAMPLE_ENTRY_NOT_OBJECT",
    "_SAMPLE_MISSING_TOKEN_FIELDS",
    "_SAMPLE_ENTRY_VERSION_MISMATCH",
    "_SAMPLE_BAD_LOGPROB_LENGTH",
    "_TRAIN_MISSING_DEDUP",
    "_SAVE_NON_OBJECT_PAYLOAD",
    "_SAVE_MISSING_REF",
    "_LOAD_MISSING_CURRENT_VERSION",
    "_READBACK_BAD_CURRENT_VERSION_TYPE",
    "_READBACK_BAD_VISIBLE_VERSION_TYPE",
    "_READBACK_VISIBLE_AHEAD",
    "_READBACK_BAD_APPLIED_REQUESTS_TYPE",
    "_READBACK_NONCONTIGUOUS_REQUESTS",
    "_READBACK_MISMATCHED_CURRENT_VERSION",
    "_READBACK_BAD_REQUEST_ID_TYPE",
    "_READBACK_BAD_REQUEST_VERSION_TYPE",
    "_READBACK_BAD_CHECKPOINT_NAME_TYPE",
)


def _enable_drop_train_ack() -> None:
    global _DROP_TRAIN_ACK
    _DROP_TRAIN_ACK = True


def _disable_drop_train_ack() -> None:
    global _DROP_TRAIN_ACK
    _DROP_TRAIN_ACK = False


def _set_protocol_flag(flag_name: str, enabled: bool) -> None:
    globals()[flag_name] = enabled


def _reset_protocol_flags() -> None:
    for flag_name in _PROTOCOL_FLAG_NAMES:
        _set_protocol_flag(flag_name, False)


def _make_protocol_flag_fault(name: str, flag_name: str) -> LambdaFault:
    return LambdaFault(
        name,
        lambda flag_name=flag_name: _set_protocol_flag(flag_name, True),
        lambda flag_name=flag_name: _set_protocol_flag(flag_name, False),
    )


def _json_dumps(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True)


def _parse_json_object(raw: object, *, op: str) -> dict[str, object]:
    if not isinstance(raw, str) or not raw:
        raise BackendProtocolError(
            f"{op} returned empty or non-string payload: {type(raw).__name__}"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BackendProtocolError(f"{op} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise BackendProtocolError(f"{op} returned non-object JSON payload")
    return payload


def _require_int(payload: dict[str, object], key: str, *, op: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BackendProtocolError(f"{op} missing integer field '{key}'")
    return value


def _require_bool(payload: dict[str, object], key: str, *, op: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise BackendProtocolError(f"{op} missing bool field '{key}'")
    return value


def _require_str(payload: dict[str, object], key: str, *, op: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise BackendProtocolError(f"{op} missing string field '{key}'")
    return value


@dataclass
class _BackendService:
    """Small deterministic backend model with explicit policy visibility."""

    current_version: int = 0
    visible_version: int = 0
    applied_requests: dict[str, int] = field(default_factory=dict)
    last_checkpoint_name: str = ""

    def checkpoint(self, name: str) -> dict[str, object]:
        self.last_checkpoint_name = name
        self.visible_version = self.current_version
        return {
            "checkpoint": name,
            "visible_version": self.visible_version,
        }

    def sample(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, object]:
        groups: list[list[dict[str, object]]] = []
        token_cap = max(1, min(max_tokens, 4))
        for prompt_idx, prompt_ids in enumerate(prompt_ids_list):
            group: list[dict[str, object]] = []
            for sample_idx in range(num_samples):
                token_count = min(token_cap, 1 + ((prompt_idx + sample_idx) % token_cap))
                token_base = self.visible_version * 1000 + prompt_idx * 100 + sample_idx * 10
                token_ids = [token_base + i for i in range(token_count)]
                logprobs = [-(0.1 + 0.01 * i) for i in range(token_count)]
                group.append(
                    {
                        "token_ids": token_ids,
                        "logprobs": logprobs,
                        "version": self.visible_version,
                        "prompt_len": len(prompt_ids),
                    }
                )
            groups.append(group)
        return {
            "version": self.visible_version,
            "checkpoint": self.last_checkpoint_name,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "groups": groups,
        }

    def train(
        self,
        request_id: str,
        datums: list[dict[str, object]],
        lr: float,
        weight_decay: float,
    ) -> dict[str, object]:
        dedup = request_id in self.applied_requests
        if dedup:
            applied_version = self.applied_requests[request_id]
        else:
            self.current_version += 1
            applied_version = self.current_version
            self.applied_requests[request_id] = applied_version
        return {
            "request_id": request_id,
            "applied_version": applied_version,
            "dedup": dedup,
            "num_datums": len(datums),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        }

    def export_state(self) -> dict[str, object]:
        return {
            "current_version": self.current_version,
            "visible_version": self.visible_version,
            "applied_requests": dict(self.applied_requests),
            "last_checkpoint_name": self.last_checkpoint_name,
        }

    def import_state(self, payload: dict[str, object]) -> None:
        current_version = payload.get("current_version")
        visible_version = payload.get("visible_version")
        applied_requests = payload.get("applied_requests")
        checkpoint_name = payload.get("last_checkpoint_name", "")

        if not isinstance(current_version, int) or isinstance(current_version, bool):
            raise ValueError("current_version must be an integer")
        if not isinstance(visible_version, int) or isinstance(visible_version, bool):
            raise ValueError("visible_version must be an integer")
        if visible_version > current_version:
            raise ValueError("visible_version cannot exceed current_version")
        if not isinstance(applied_requests, dict):
            raise ValueError("applied_requests must be an object")
        if not isinstance(checkpoint_name, str):
            raise ValueError("last_checkpoint_name must be a string")

        normalized_requests: dict[str, int] = {}
        for key, value in applied_requests.items():
            if not isinstance(key, str):
                raise ValueError("request ids must be strings")
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("applied request versions must be integers")
            normalized_requests[key] = value

        versions = sorted(normalized_requests.values())
        if versions and versions != list(range(1, max(versions) + 1)):
            raise ValueError("applied request versions must stay contiguous")
        if versions and max(versions) != current_version:
            raise ValueError("current_version must match the latest applied request")

        self.current_version = current_version
        self.visible_version = visible_version
        self.applied_requests = normalized_requests
        self.last_checkpoint_name = checkpoint_name


def _write_state_slot(store: dict[str, str], ref: str, text: str) -> str:
    store[ref] = text
    return ref


def _read_state_slot(store: dict[str, str], ref: str) -> str:
    return store[ref]


def _transport_checkpoint(service: _BackendService, name: str) -> str:
    payload = service.checkpoint(name)
    if _CHECKPOINT_MISSING_VISIBLE_VERSION:
        payload = dict(payload)
        payload.pop("visible_version", None)
    if _CHECKPOINT_EMPTY_PAYLOAD:
        return ""
    return _json_dumps(payload)


def _transport_sample(
    service: _BackendService,
    prompt_ids_list: list[list[int]],
    num_samples: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    payload = service.sample(prompt_ids_list, num_samples, max_tokens, temperature, top_p)
    if _SAMPLE_MISSING_GROUPS:
        payload = dict(payload)
        payload.pop("groups", None)
    if _SAMPLE_WRONG_GROUP_COUNT:
        payload = dict(payload)
        groups = payload.get("groups")
        if isinstance(groups, list):
            payload["groups"] = groups[:-1]
    if _SAMPLE_GROUP_NOT_LIST:
        payload = dict(payload)
        groups = payload.get("groups")
        if isinstance(groups, list) and groups:
            payload["groups"] = ["not-a-group", *groups[1:]]
    if _SAMPLE_WRONG_COMPLETION_COUNT:
        payload = dict(payload)
        groups = payload.get("groups")
        if isinstance(groups, list) and groups and isinstance(groups[0], list):
            payload["groups"] = [groups[0][:-1], *groups[1:]]
    if _SAMPLE_ENTRY_NOT_OBJECT:
        payload = dict(payload)
        groups = payload.get("groups")
        if isinstance(groups, list) and groups and isinstance(groups[0], list) and groups[0]:
            payload["groups"] = [[123, *groups[0][1:]], *groups[1:]]
    if _SAMPLE_MISSING_TOKEN_FIELDS:
        payload = dict(payload)
        groups = payload.get("groups")
        if isinstance(groups, list) and groups and isinstance(groups[0], list) and groups[0]:
            first_entry = groups[0][0]
            if isinstance(first_entry, dict):
                first_entry = dict(first_entry)
                first_entry.pop("token_ids", None)
                first_entry.pop("logprobs", None)
                groups[0][0] = first_entry
    if _SAMPLE_ENTRY_VERSION_MISMATCH:
        payload = dict(payload)
        groups = payload.get("groups")
        if isinstance(groups, list) and groups and isinstance(groups[0], list) and groups[0]:
            first_entry = groups[0][0]
            if isinstance(first_entry, dict):
                first_entry = dict(first_entry)
                first_entry["version"] = int(payload["version"]) + 1
                groups[0][0] = first_entry
    if _SAMPLE_BAD_LOGPROB_LENGTH:
        payload = dict(payload)
        groups = payload.get("groups")
        if isinstance(groups, list) and groups and isinstance(groups[0], list) and groups[0]:
            first_entry = groups[0][0]
            if isinstance(first_entry, dict):
                first_entry = dict(first_entry)
                logprobs = first_entry.get("logprobs")
                if isinstance(logprobs, list) and logprobs:
                    first_entry["logprobs"] = logprobs[:-1]
                    groups[0][0] = first_entry
    if _SAMPLE_INVALID_JSON:
        return "{"
    return _json_dumps(payload)


def _transport_train(
    service: _BackendService,
    request_id: str,
    datums: list[dict[str, object]],
    lr: float,
    weight_decay: float,
) -> str:
    payload = service.train(request_id, datums, lr, weight_decay)
    if _DROP_TRAIN_ACK:
        raise ConnectionError("Simulated post-commit ACK loss")
    if _TRAIN_MISSING_DEDUP:
        payload = dict(payload)
        payload.pop("dedup", None)
    return _json_dumps(payload)


def _transport_save(
    service: _BackendService,
    store: dict[str, str],
    ref: str,
) -> str:
    state_text = _json_dumps(service.export_state())
    saved_ref = _write_state_slot(store, ref, state_text)
    if _SAVE_NON_OBJECT_PAYLOAD:
        return "[]"
    payload: dict[str, object] = {"ref": saved_ref, "version": service.current_version}
    if _SAVE_MISSING_REF:
        payload.pop("ref", None)
    return _json_dumps(payload)


def _transport_load(
    service: _BackendService,
    store: dict[str, str],
    ref: str,
) -> str:
    state_text = _read_state_slot(store, ref)
    payload = _parse_json_object(state_text, op="load_state/readback")
    if _READBACK_BAD_CURRENT_VERSION_TYPE:
        payload = dict(payload)
        payload["current_version"] = "bad-version"
    if _READBACK_BAD_VISIBLE_VERSION_TYPE:
        payload = dict(payload)
        payload["visible_version"] = "bad-visible-version"
    if _READBACK_VISIBLE_AHEAD:
        payload = dict(payload)
        current_version = payload.get("current_version", 0)
        if isinstance(current_version, int) and not isinstance(current_version, bool):
            payload["visible_version"] = current_version + 1
    if _READBACK_BAD_APPLIED_REQUESTS_TYPE:
        payload = dict(payload)
        payload["applied_requests"] = ["bad-requests"]
    if _READBACK_NONCONTIGUOUS_REQUESTS:
        payload = dict(payload)
        payload["applied_requests"] = {"request-gap": 2}
        payload["current_version"] = 2
    if _READBACK_MISMATCHED_CURRENT_VERSION:
        payload = dict(payload)
        payload["applied_requests"] = {"request-0": 1}
        payload["current_version"] = 2
    if _READBACK_BAD_REQUEST_ID_TYPE:
        payload = dict(payload)
        payload["applied_requests"] = {1: 1}
    if _READBACK_BAD_REQUEST_VERSION_TYPE:
        payload = dict(payload)
        payload["applied_requests"] = {"request-0": "bad-version"}
    if _READBACK_BAD_CHECKPOINT_NAME_TYPE:
        payload = dict(payload)
        payload["last_checkpoint_name"] = 123
    service.import_state(payload)
    response: dict[str, object] = {
        "ref": ref,
        "current_version": service.current_version,
    }
    if _LOAD_MISSING_CURRENT_VERSION:
        response.pop("current_version", None)
    return _json_dumps(response)


class _RemoteBackendAdapter:
    """Reference adapter for a remote backend protocol."""

    def __init__(
        self,
        service: _BackendService,
        store: dict[str, str],
        *,
        client_id: int,
    ) -> None:
        self.service = service
        self.store = store
        self.client_id = client_id
        self.request_seq = 0
        self.last_sampled_version = 0
        self.last_loaded_ref = ""
        self.seen_dedup_retry = False

    def _next_request_id(self) -> str:
        request_id = f"client-{self.client_id}:req-{self.request_seq}"
        self.request_seq += 1
        return request_id

    def checkpoint(self, name: str) -> None:
        payload = _parse_json_object(
            _transport_checkpoint(self.service, name),
            op="checkpoint",
        )
        _require_str(payload, "checkpoint", op="checkpoint")
        _require_int(payload, "visible_version", op="checkpoint")

    def sample(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[tuple[list[int], list[float]]]]:
        payload = _parse_json_object(
            _transport_sample(
                self.service,
                prompt_ids_list,
                num_samples,
                max_tokens,
                temperature,
                top_p,
            ),
            op="sample",
        )
        version = _require_int(payload, "version", op="sample")
        groups_raw = payload.get("groups")
        if not isinstance(groups_raw, list):
            raise BackendProtocolError("sample missing groups list")
        if len(groups_raw) != len(prompt_ids_list):
            raise BackendProtocolError("sample returned wrong number of groups")

        groups: list[list[tuple[list[int], list[float]]]] = []
        for group_raw in groups_raw:
            if not isinstance(group_raw, list):
                raise BackendProtocolError("sample group is not a list")
            if len(group_raw) != num_samples:
                raise BackendProtocolError("sample returned wrong number of completions")
            group: list[tuple[list[int], list[float]]] = []
            for entry_raw in group_raw:
                if not isinstance(entry_raw, dict):
                    raise BackendProtocolError("sample entry is not an object")
                token_ids = entry_raw.get("token_ids")
                logprobs = entry_raw.get("logprobs")
                sample_version = entry_raw.get("version")
                if not isinstance(token_ids, list) or not isinstance(logprobs, list):
                    raise BackendProtocolError("sample entry missing token_ids/logprobs")
                if not isinstance(sample_version, int) or sample_version != version:
                    raise BackendProtocolError("sample entry version mismatch")
                if len(token_ids) != len(logprobs):
                    raise BackendProtocolError("sample token_ids/logprobs length mismatch")
                group.append(
                    (
                        [int(tok) for tok in token_ids],
                        [float(lp) for lp in logprobs],
                    )
                )
            groups.append(group)

        self.last_sampled_version = version
        return groups

    def train_step(
        self,
        all_tokens: list[list[int]],
        all_logprobs: list[list[float]],
        all_advantages: list[list[float]],
        lr: float,
        weight_decay: float,
    ) -> float:
        request_id = self._next_request_id()
        datums: list[dict[str, object]] = []
        for tokens, logprobs, advantages in zip(
            all_tokens, all_logprobs, all_advantages
        ):
            datums.append(
                {
                    "tokens": len(tokens),
                    "logprobs": len(logprobs),
                    "advantages": len(advantages),
                }
            )

        transport_errors = (
            BackendProtocolError,
            ConnectionError,
            TimeoutError,
            OSError,
            network.HTTPFaultError,
        )
        last_exc: Exception | None = None
        for _attempt in range(2):
            try:
                payload = _parse_json_object(
                    _transport_train(
                        self.service,
                        request_id,
                        datums,
                        lr,
                        weight_decay,
                    ),
                    op="train_step",
                )
                _require_str(payload, "request_id", op="train_step")
                applied_version = _require_int(payload, "applied_version", op="train_step")
                dedup = _require_bool(payload, "dedup", op="train_step")
                if dedup:
                    self.seen_dedup_retry = True
                return float(applied_version)
            except transport_errors as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    def save_adapter(self, path: str, name: str) -> str:
        _ = path
        payload = _parse_json_object(
            _transport_save(self.service, self.store, ref=f"{name}.json"),
            op="save_adapter",
        )
        return _require_str(payload, "ref", op="save_adapter")

    def load_state(self, ref: str) -> None:
        payload = _parse_json_object(
            _transport_load(self.service, self.store, ref),
            op="load_state",
        )
        self.last_loaded_ref = _require_str(payload, "ref", op="load_state")
        _require_int(payload, "current_version", op="load_state")


class BackendWorkflowChaos(ChaosTest):
    """Stateful backend protocol testing under transport faults."""

    swarm = True
    faults = [
        timing.timeout(f"{__name__}._transport_checkpoint"),
        network.intermittent_http_error(
            f"{__name__}._transport_sample",
            every_n=3,
            status_code=503,
        ),
        network.connection_reset(f"{__name__}._transport_train"),
        LambdaFault("drop_train_ack", _enable_drop_train_ack, _disable_drop_train_ack),
        io_faults.error_on_call(f"{__name__}._write_state_slot", IOError),
        io_faults.truncate_output(f"{__name__}._read_state_slot", fraction=0.5),
        _make_protocol_flag_fault(
            "checkpoint_empty_payload",
            "_CHECKPOINT_EMPTY_PAYLOAD",
        ),
        _make_protocol_flag_fault(
            "checkpoint_missing_visible_version",
            "_CHECKPOINT_MISSING_VISIBLE_VERSION",
        ),
        _make_protocol_flag_fault(
            "sample_invalid_json",
            "_SAMPLE_INVALID_JSON",
        ),
        _make_protocol_flag_fault(
            "sample_missing_groups",
            "_SAMPLE_MISSING_GROUPS",
        ),
        _make_protocol_flag_fault(
            "sample_wrong_group_count",
            "_SAMPLE_WRONG_GROUP_COUNT",
        ),
        _make_protocol_flag_fault(
            "sample_group_not_list",
            "_SAMPLE_GROUP_NOT_LIST",
        ),
        _make_protocol_flag_fault(
            "sample_wrong_completion_count",
            "_SAMPLE_WRONG_COMPLETION_COUNT",
        ),
        _make_protocol_flag_fault(
            "sample_entry_not_object",
            "_SAMPLE_ENTRY_NOT_OBJECT",
        ),
        _make_protocol_flag_fault(
            "sample_missing_token_fields",
            "_SAMPLE_MISSING_TOKEN_FIELDS",
        ),
        _make_protocol_flag_fault(
            "sample_entry_version_mismatch",
            "_SAMPLE_ENTRY_VERSION_MISMATCH",
        ),
        _make_protocol_flag_fault(
            "sample_bad_logprob_length",
            "_SAMPLE_BAD_LOGPROB_LENGTH",
        ),
        _make_protocol_flag_fault(
            "train_missing_dedup",
            "_TRAIN_MISSING_DEDUP",
        ),
        _make_protocol_flag_fault(
            "save_non_object_payload",
            "_SAVE_NON_OBJECT_PAYLOAD",
        ),
        _make_protocol_flag_fault(
            "save_missing_ref",
            "_SAVE_MISSING_REF",
        ),
        _make_protocol_flag_fault(
            "load_missing_current_version",
            "_LOAD_MISSING_CURRENT_VERSION",
        ),
        _make_protocol_flag_fault(
            "readback_bad_current_version_type",
            "_READBACK_BAD_CURRENT_VERSION_TYPE",
        ),
        _make_protocol_flag_fault(
            "readback_bad_visible_version_type",
            "_READBACK_BAD_VISIBLE_VERSION_TYPE",
        ),
        _make_protocol_flag_fault(
            "readback_visible_ahead",
            "_READBACK_VISIBLE_AHEAD",
        ),
        _make_protocol_flag_fault(
            "readback_bad_applied_requests_type",
            "_READBACK_BAD_APPLIED_REQUESTS_TYPE",
        ),
        _make_protocol_flag_fault(
            "readback_noncontiguous_requests",
            "_READBACK_NONCONTIGUOUS_REQUESTS",
        ),
        _make_protocol_flag_fault(
            "readback_mismatched_current_version",
            "_READBACK_MISMATCHED_CURRENT_VERSION",
        ),
        _make_protocol_flag_fault(
            "readback_bad_request_id_type",
            "_READBACK_BAD_REQUEST_ID_TYPE",
        ),
        _make_protocol_flag_fault(
            "readback_bad_request_version_type",
            "_READBACK_BAD_REQUEST_VERSION_TYPE",
        ),
        _make_protocol_flag_fault(
            "readback_bad_checkpoint_name_type",
            "_READBACK_BAD_CHECKPOINT_NAME_TYPE",
        ),
    ]

    def __init__(self) -> None:
        super().__init__()
        _disable_drop_train_ack()
        _reset_protocol_flags()
        self.service = _BackendService()
        self.store: dict[str, str] = {}
        self.client_id = 0
        self.adapter = _RemoteBackendAdapter(
            self.service,
            self.store,
            client_id=self.client_id,
        )
        self.checkpoint_calls = 0
        self.save_calls = 0
        self.saved_refs: list[str] = []
        self.seen_checkpoint = False
        self.seen_train = False
        self.seen_sample = False
        self.seen_save = False
        self.seen_load = False
        self.seen_stale_sample = False
        self.protocol_error_kinds: set[str] = set()
        self.state_validation_error_kinds: set[str] = set()

    def _rebind_adapter(self) -> None:
        """Restore adapter aliasing after explorer checkpoint restores."""

        prior = self.adapter
        rebound = _RemoteBackendAdapter(
            self.service,
            self.store,
            client_id=self.client_id,
        )
        rebound.request_seq = prior.request_seq
        rebound.last_sampled_version = prior.last_sampled_version
        rebound.last_loaded_ref = prior.last_loaded_ref
        rebound.seen_dedup_retry = prior.seen_dedup_retry
        self.adapter = rebound

    @staticmethod
    def _bounded_int(value: int, *, lower: int, upper: int) -> bool:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and lower <= value <= upper
        )

    def _record_protocol_error(self, exc: BackendProtocolError) -> None:
        message = str(exc)
        if "empty or non-string payload" in message:
            self.protocol_error_kinds.add("empty_or_non_string_payload")
            return
        if "invalid JSON" in message:
            self.protocol_error_kinds.add("invalid_json")
            return
        if "non-object JSON payload" in message:
            self.protocol_error_kinds.add("non_object_payload")
            return
        if "missing integer field" in message:
            self.protocol_error_kinds.add("missing_integer_field")
            return
        if "missing bool field" in message:
            self.protocol_error_kinds.add("missing_bool_field")
            return
        if "missing string field" in message:
            self.protocol_error_kinds.add("missing_string_field")
            return
        self.protocol_error_kinds.add("schema_mismatch")

    def _record_state_validation_error(self, exc: ValueError) -> None:
        message = str(exc)
        if "current_version must be an integer" in message:
            self.state_validation_error_kinds.add("bad_current_version_type")
            return
        if "visible_version must be an integer" in message:
            self.state_validation_error_kinds.add("bad_visible_version_type")
            return
        if "visible_version cannot exceed current_version" in message:
            self.state_validation_error_kinds.add("visible_version_ahead")
            return
        if "applied_requests must be an object" in message:
            self.state_validation_error_kinds.add("bad_applied_requests_type")
            return
        if "last_checkpoint_name must be a string" in message:
            self.state_validation_error_kinds.add("bad_checkpoint_name_type")
            return
        if "request ids must be strings" in message:
            self.state_validation_error_kinds.add("bad_request_id_type")
            return
        if "applied request versions must be integers" in message:
            self.state_validation_error_kinds.add("bad_request_version_type")
            return
        if "applied request versions must stay contiguous" in message:
            self.state_validation_error_kinds.add("noncontiguous_request_versions")
            return
        if "current_version must match the latest applied request" in message:
            self.state_validation_error_kinds.add("mismatched_current_version")
            return
        self.state_validation_error_kinds.add("state_validation_error")

    def state_hash(self) -> int:
        self._rebind_adapter()
        return hash(
            (
                min(self.service.current_version, 8),
                min(self.service.visible_version, 8),
                self.service.visible_version < self.service.current_version,
                len(self.saved_refs) > 0,
                len(self.saved_refs) > 2,
                self.adapter.seen_dedup_retry,
                bool(self.adapter.last_loaded_ref),
                bool(self.protocol_error_kinds),
                len(self.protocol_error_kinds) >= 3,
                bool(self.state_validation_error_kinds),
                len(self.state_validation_error_kinds) >= 2,
                self.client_id % 4,
            )
        )

    @rule()
    def checkpoint(self) -> None:
        self._rebind_adapter()
        self.checkpoint_calls += 1
        try:
            self.adapter.checkpoint(f"step_{self.checkpoint_calls}")
        except BackendProtocolError as exc:
            self._record_protocol_error(exc)
            return
        except (TimeoutError, OSError):
            return
        self.seen_checkpoint = True
        always(
            self.service.visible_version == self.service.current_version,
            "checkpoint publishes the current backend policy",
        )

    @rule(
        datum_count=st.integers(min_value=1, max_value=3),
        token_count=st.integers(min_value=1, max_value=4),
    )
    def train(self, datum_count: int, token_count: int) -> None:
        self._rebind_adapter()
        if not self._bounded_int(datum_count, lower=1, upper=3):
            return
        if not self._bounded_int(token_count, lower=1, upper=4):
            return
        all_tokens: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        all_advantages: list[list[float]] = []
        for datum_idx in range(datum_count):
            prompt = [100 + datum_idx]
            completion = [200 + i for i in range(token_count)]
            full_tokens = prompt + completion
            all_tokens.append(full_tokens)
            all_logprobs.append([0.0] + [-0.2] * token_count)
            all_advantages.append([0.0] + [1.0] * token_count)

        try:
            version = self.adapter.train_step(
                all_tokens,
                all_logprobs,
                all_advantages,
                lr=1e-4,
                weight_decay=0.0,
            )
        except BackendProtocolError as exc:
            self._record_protocol_error(exc)
            return
        except (ConnectionError, TimeoutError, OSError, network.HTTPFaultError):
            return

        self.seen_train = True
        always(version >= 1.0, "successful train_step returns a versioned update")
        if self.adapter.seen_dedup_retry:
            reachable("deduplicated_retry_after_ambiguous_train_failure")

    @rule(
        num_prompts=st.integers(min_value=1, max_value=3),
        num_samples=st.integers(min_value=1, max_value=3),
    )
    def sample(self, num_prompts: int, num_samples: int) -> None:
        self._rebind_adapter()
        if not self._bounded_int(num_prompts, lower=1, upper=3):
            return
        if not self._bounded_int(num_samples, lower=1, upper=3):
            return
        prompts = [[11 + i, 21 + i] for i in range(num_prompts)]
        try:
            groups = self.adapter.sample(
                prompts,
                num_samples,
                max_tokens=4,
                temperature=0.7,
                top_p=0.95,
            )
        except BackendProtocolError as exc:
            self._record_protocol_error(exc)
            return
        except (ConnectionError, TimeoutError, OSError, network.HTTPFaultError):
            return

        self.seen_sample = True
        always(len(groups) == num_prompts, "one group per prompt")
        for group in groups:
            always(len(group) == num_samples, "one completion per requested sample")
        always(
            self.adapter.last_sampled_version == self.service.visible_version,
            "sampling sees only checkpointed policy weights",
        )
        if self.adapter.last_sampled_version < self.service.current_version:
            self.seen_stale_sample = True
            reachable("sample_version_lags_until_checkpoint")

    @rule()
    def train_then_sample_without_checkpoint(self) -> None:
        self._rebind_adapter()
        all_tokens = [[1, 2, 3]]
        all_logprobs = [[0.0, -0.2, -0.2]]
        all_advantages = [[0.0, 1.0, 1.0]]
        try:
            self.adapter.train_step(
                all_tokens,
                all_logprobs,
                all_advantages,
                lr=1e-4,
                weight_decay=0.0,
            )
            self.adapter.sample(
                [[9, 9]],
                num_samples=1,
                max_tokens=2,
                temperature=0.7,
                top_p=0.95,
            )
        except BackendProtocolError as exc:
            self._record_protocol_error(exc)
            return
        except (ConnectionError, TimeoutError, OSError, network.HTTPFaultError):
            return

        if self.service.current_version > self.service.visible_version:
            self.seen_stale_sample = True
            always(
                self.adapter.last_sampled_version == self.service.visible_version,
                "sampling must not observe uncheckpointed updates",
            )
            reachable("sample_version_lags_until_checkpoint")

    @rule()
    def save_snapshot(self) -> None:
        self._rebind_adapter()
        self.save_calls += 1
        try:
            ref = self.adapter.save_adapter("unused", f"snapshot_{self.save_calls}")
        except BackendProtocolError as exc:
            self._record_protocol_error(exc)
            return
        except (OSError, IOError):
            return

        self.seen_save = True
        self.saved_refs.append(ref)
        always(ref in self.store, "save returns a persisted snapshot ref")

    @rule(which=st.integers(min_value=0, max_value=8))
    def load_snapshot(self, which: int) -> None:
        self._rebind_adapter()
        if not self.saved_refs:
            return
        ref = self.saved_refs[which % len(self.saved_refs)]
        try:
            self.adapter.load_state(ref)
        except BackendProtocolError as exc:
            self._record_protocol_error(exc)
            return
        except ValueError as exc:
            self._record_state_validation_error(exc)
            return
        except (
            KeyError,
            OSError,
            IOError,
        ):
            return

        self.seen_load = True
        always(
            self.adapter.last_loaded_ref == ref,
            "load_state records the successfully restored reference",
        )
        reachable("save_load_roundtrip")

    @rule()
    def restart_adapter(self) -> None:
        self._rebind_adapter()
        self.client_id += 1
        self.adapter = _RemoteBackendAdapter(
            self.service,
            self.store,
            client_id=self.client_id,
        )
        reachable("adapter_restart")

    @invariant()
    def visible_version_never_ahead(self) -> None:
        self._rebind_adapter()
        assert self.service.visible_version <= self.service.current_version

    @invariant()
    def applied_request_versions_stay_contiguous(self) -> None:
        self._rebind_adapter()
        versions = sorted(self.service.applied_requests.values())
        if versions:
            assert versions == list(range(1, max(versions) + 1))
            assert max(versions) == self.service.current_version
        else:
            assert self.service.current_version == 0

    @invariant()
    def saved_snapshots_are_valid_json(self) -> None:
        self._rebind_adapter()
        for ref in self.saved_refs:
            if ref not in self.store:
                continue
            payload = json.loads(self.store[ref])
            assert isinstance(payload, dict)
            assert payload["visible_version"] <= payload["current_version"]

    @invariant()
    def adapter_points_at_live_machine_state(self) -> None:
        self._rebind_adapter()
        assert self.adapter.service is self.service
        assert self.adapter.store is self.store

    def teardown(self) -> None:
        self._rebind_adapter()
        sometimes(self.seen_train, "training path is exercised")
        sometimes(self.seen_checkpoint, "checkpoint path is exercised")
        sometimes(self.seen_save, "save path is exercised")
        sometimes(self.seen_sample, "sampling path is exercised")
        sometimes(
            bool(self.protocol_error_kinds),
            "protocol corruption paths are exercised",
        )
        sometimes(
            bool(self.state_validation_error_kinds),
            "persisted state validation paths are exercised",
        )
        if self.saved_refs:
            sometimes(self.seen_load, "load path is exercised after saving")
        sometimes(
            self.seen_stale_sample,
            "sampling sometimes lags behind training until checkpoint",
        )
        _reset_protocol_flags()
        _disable_drop_train_ack()
        super().teardown()


TestBackendWorkflowChaos = BackendWorkflowChaos.TestCase
TestBackendWorkflowChaos.settings = settings(
    max_examples=40,
    stateful_step_count=20,
    deadline=None,
)
