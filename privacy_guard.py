import hashlib
import json


def verify_weight_payload(state_dict, allowed_keys):
    payload_keys = set(state_dict.keys())
    expected_keys = set(allowed_keys)
    if payload_keys != expected_keys:
        raise ValueError("Unexpected tensor keys detected in client update")
    for key, tensor in state_dict.items():
        if tensor.ndim == 0 and tensor.numel() == 1:
            continue
        if tensor.numel() > 500000:
            raise ValueError(f"Oversized tensor blocked on {key}")
    return True


def fingerprint_weights(state_dict):
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        digest.update(key.encode("utf-8"))
        digest.update(state_dict[key].cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def build_audit_record(round_num, node_name, sample_count, state_dict):
    return json.dumps({
        "round": round_num,
        "node": node_name,
        "samples": int(sample_count),
        "weight_hash": fingerprint_weights(state_dict),
        "raw_data_shared": False,
    })
