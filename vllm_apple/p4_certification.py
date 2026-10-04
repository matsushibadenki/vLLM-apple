"""Read-only release certification gate; report hashes are integrity, not signatures."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

MAX_BYTES = 4 * 1024 * 1024
ROLES = ("quality", "performance", "soak", "recovery", "rollback", "installation",
         "uninstallation", "dependencies", "licenses")
IDENTITY_FIELDS = ("hardware", "os", "model", "backend", "configuration")
RECOVERY_CASES = ("os_update", "backend_update", "startup_failure", "profile_corruption")
MESSAGES = {
    "certified": {"en": "Certified for this exact configuration.",
                  "ja": "この構成で認定済みです。", "zh": "此配置已通过认证。"},
    "uncertified": {"en": "Certification evidence is missing or failed; retain the baseline.",
                    "ja": "認定証拠が不足または不合格のため、基準経路を保持します。",
                    "zh": "认证证据缺失或未通过，保留基线路径。"},
    "resource_pressure": {"en": "Thermal or memory pressure prevents profile activation.",
                          "ja": "熱またはメモリの制約によりprofileを有効化できません。",
                          "zh": "温度或内存压力阻止启用profile。"},
}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _hash(value: object, length: int = 64) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        c in "0123456789abcdef" for c in value)


def _load(path: Path) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("evidence must be a bounded regular file")
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError("evidence exceeds size limit")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("evidence must contain a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _evidence(root: Path, reference: dict) -> tuple[dict, str]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("invalid evidence reference")
    path = reference["path"]
    if (not isinstance(path, str) or not path or "\\" in path
            or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
            or not _hash(reference["sha256"])):
        raise ValueError("unsafe evidence reference")
    root = root.resolve()
    target = root / path
    # Do not follow symlinks, including directory components.
    components = (root.joinpath(*PurePosixPath(path).parts[:i])
                  for i in range(1, len(PurePosixPath(path).parts) + 1))
    if any(p.is_symlink() for p in components):
        raise ValueError("evidence symlink is forbidden")
    target.resolve().relative_to(root.resolve())
    payload, sha = _load(target)
    if sha != reference["sha256"]:
        raise ValueError("evidence hash mismatch")
    return payload, sha


def verify_certification(bundle_path: Path, *, source_commit: str,
                         artifact_sha256: str) -> dict:
    """Verify all advertised matrix cells. No partial-matrix release certification."""
    bundle, bundle_sha = _load(bundle_path)
    if (bundle.get("schema_version") != 1 or not _hash(source_commit, 40)
            or not _hash(artifact_sha256)
            or bundle.get("source_commit") != source_commit
            or bundle.get("artifact_sha256") != artifact_sha256):
        raise ValueError("certification source or artifact mismatch")
    cells = bundle.get("cells")
    if not isinstance(cells, list) or not 1 <= len(cells) <= 64:
        raise ValueError("certification matrix must contain 1..64 cells")
    results = []
    identities = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("invalid certification cell")
        identity = cell.get("identity")
        if (not isinstance(identity, dict) or set(identity) != set(IDENTITY_FIELDS)
                or any(not _hash(v) for v in identity.values())):
            raise ValueError("cell requires five exact identity hashes")
        scope = digest(identity)
        if scope in identities:
            raise ValueError("duplicate certification cell")
        identities.add(scope)
        references = cell.get("evidence")
        if not isinstance(references, dict) or set(references) != set(ROLES):
            raise ValueError("cell must reference every required evidence role")
        reasons, hashes = [], {}
        for role in ROLES:
            report, sha = _evidence(bundle_path.parent, references[role])
            hashes[role] = sha
            if (report.get("schema_version") != 1 or report.get("role") != role
                    or report.get("scope_sha256") != scope
                    or report.get("source_commit") != source_commit
                    or report.get("artifact_sha256") != artifact_sha256):
                reasons.append(role + ":identity_mismatch")
            if report.get("passed") is not True:
                reasons.append(role + ":failed_or_unverified")
            # Every envelope binds the actual raw collector output, not just a summary bool.
            raw, raw_sha = _evidence(bundle_path.parent, report.get("raw_evidence"))
            hashes[role + "_raw"] = raw_sha
            if role != "performance" and raw.get("passed") is not True:
                reasons.append(role + ":raw_evidence_failed")
            if role == "soak":
                duration = raw.get("duration_seconds")
                if (type(duration) not in (int, float) or not math.isfinite(duration)
                        or duration < 86400 or raw.get("clean_shutdown") is not True):
                    reasons.append("soak:24h_or_clean_shutdown_missing")
            if role == "quality":
                slices = raw.get("slices")
                required = ("en", "ja", "zh", "coding", "tool", "long_context")
                if (not isinstance(slices, dict)
                        or any(slices.get(s) is not True for s in required)):
                    reasons.append("quality:required_slice_failed_or_missing")
            if role == "performance":
                if (raw.get("independent_acquisition_verified") is not True
                        or raw.get("standard_adoption_eligible") is not True
                        or raw.get("baseline_retained") is not False
                        or not isinstance(raw.get("policy"), dict)
                        or raw["policy"].get("scope_sha256") != scope
                        or raw.get("report_id") != digest(
                            {k: v for k, v in raw.items() if k != "report_id"})):
                    reasons.append("performance:p3_gate_unqualified")
            if role == "recovery":
                cases = raw.get("cases")
                if (not isinstance(cases, dict)
                        or any(cases.get(case) is not True for case in RECOVERY_CASES)):
                    reasons.append("recovery:required_case_failed_or_missing")
            if role == "rollback":
                if (not _hash(raw.get("restored_profile_sha256"))
                        or raw.get("inference_verified") is not True):
                    reasons.append("rollback:restoration_unverified")
            if role in ("installation", "uninstallation"):
                if raw.get("independent_clean_machine_verified") is not True:
                    reasons.append(role + ":clean_machine_unverified")
            if role == "dependencies" and raw.get("lock_verified") is not True:
                reasons.append("dependencies:lock_unverified")
            if role == "licenses" and raw.get("sbom_license_review_verified") is not True:
                reasons.append("licenses:sbom_review_unverified")
        results.append(dict(scope_sha256=scope, identity=identity, certified=not reasons,
                            rejection_reasons=reasons, evidence_sha256=hashes))
    report = dict(schema_version=1, source_commit=source_commit,
                  artifact_sha256=artifact_sha256, bundle_sha256=bundle_sha,
                  passed=all(c["certified"] for c in results), cells=results,
                  automatic_application=False)
    report["report_id"] = digest(report)
    return report


def profile_activation(report: dict, identity: dict, *, thermal_safe: bool,
                       memory_safe: bool) -> dict:
    """Choose an activation status only; caller retains ownership of model and state."""
    if thermal_safe is not True or memory_safe is not True:
        code = "resource_pressure"
    elif (report.get("passed") is True and report.get("report_id") == digest(
            {k: v for k, v in report.items() if k != "report_id"})
          and any(c.get("identity") == identity and c.get("certified") is True
                  for c in report.get("cells", []))):
        code = "certified"
    else:
        code = "uncertified"
    return dict(activate=code == "certified", code="p4_" + code, messages=MESSAGES[code])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--artifact-sha256", required=True)
    args = parser.parse_args()
    try:
        report = verify_certification(args.bundle, source_commit=args.source_commit,
                                      artifact_sha256=args.artifact_sha256)
    except (ValueError, OSError, TypeError) as error:
        print(json.dumps(dict(passed=False, code="p4_invalid_evidence", detail=str(error))))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
