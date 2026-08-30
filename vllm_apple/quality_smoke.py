from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .phase_probe import PhaseProbeConfig, StreamProbeResult, measure_stream

QUALITY_SMOKE_SCHEMA_VERSION = 1
QUALITY_CASES = (
    ("english", "Answer with exactly this token and nothing else: cobalt", "cobalt"),
    ("japanese", "次の語だけを出力してください。説明は不要です：青", "青"),
    ("simplified_chinese", "只输出下列汉字，不要解释：蓝", "蓝"),
)


QualityMeasure = Callable[..., StreamProbeResult]


def run_serving_quality_smoke(
    config: PhaseProbeConfig,
    *,
    measure: QualityMeasure = measure_stream,
) -> dict[str, object]:
    checks: dict[str, bool] = {}
    for name, prompt, expected in QUALITY_CASES:
        result = measure(
            replace(config, prompt=prompt, maximum_output_tokens=8, samples=1),
            expected_text=expected,
            expected_match_mode="exact",
        )
        checks[name] = result.expected_text_matched is True
    return {
        "schema_version": QUALITY_SMOKE_SCHEMA_VERSION,
        "sample_count": len(checks),
        "checks": checks,
        "stores_generated_text": False,
        "passed": all(checks.values()),
    }
