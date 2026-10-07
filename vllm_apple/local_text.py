"""Explicit local text preview for the measured M4/Gemma2 configuration."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from .hardware import detect_hardware
from .mlx_efficiency import settings

MODEL_HASHES = {
    'config.json': '41c1077a8a8b14f3e016c0000365aae99fb9eb128596c8378a178f717dca1640',
    'model.safetensors': 'f87c0f8cfa7bea0d01266bd04fae9b60babfa21a57eefbbaf5354321a0dabbf2',
    'model.safetensors.index.json': 'cb1ab8d56b40451421668d828f8650cbf17c2d901d6f43a16f9a0f3ab49e42c9',
    'tokenizer.json': '3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922',
    'tokenizer_config.json': 'f3a9ecd05833ba49de8432fff27b66bb061ad6a69d17df158de03dc07420e02a',
    'tokenizer.model': '61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2',
    'special_tokens_map.json': 'db82f8bd9b25d14f9c788e6bde64de84d42f1c2538f1c245ba6cb3e872d14b18',
}
SOURCE_HASHES = {
    'mlx_lm/models/gemma2.py': '64b0935b06fe2c4d5d4ed23a9cf62deb6218c55a88b9403a657afe9e2be8f251',
    'mlx_lm/server.py': '8514178d18ee7e5edd1db8b8077ab97079ec72d1db666b85c110fb33cdc55147',
    'mlx_lm/generate.py': '5a57043b5a6497450bce14447db3caf570ffdd22adaab34e659b31aceddda522',
}
MESSAGES = {
    'en': 'Local text preview only. Long-run stability and performance remain unqualified.',
    'ja': 'ローカルtext previewです。長時間安定性と性能は未認定です。',
    'zh': '仅为本地text preview，长时间稳定性及性能尚未认证。',
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def preview_plan(model: Path, port: int, efficiency: str = 'baseline') -> dict:
    policy = settings(efficiency)
    if not 1024 <= port <= 65535:
        raise ValueError('port must be 1024–65535')
    model = model.resolve(strict=True)
    if not model.is_dir():
        raise ValueError('a local model directory is required')
    hardware = detect_hardware()
    reasons = []
    if (not hardware.is_apple_silicon or hardware.soc != 'Apple M4'
            or hardware.memory.total_bytes < 32 * 1024**3):
        reasons.append('hardware_outside_measured_preview')
    if hardware.memory.available_bytes < 4 * 1024**3:
        reasons.append('insufficient_available_memory')
    packages = {}
    for name, required in (('mlx', '0.32.1'), ('mlx-lm', '0.32.0'), ('transformers', '5.17.0')):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
        if packages[name] != required:
            reasons.append('unreviewed_' + name)
    source_hashes = {}
    if packages.get('mlx-lm') == '0.32.0':
        distribution = importlib.metadata.distribution('mlx-lm')
        for name, expected in SOURCE_HASHES.items():
            path = Path(distribution.locate_file(name))
            source_hashes[name] = file_hash(path) if path.is_file() else None
            if source_hashes[name] != expected:
                reasons.append('dependency_source_mismatch:' + name)
    expected_weights = {name for name in MODEL_HASHES if name.endswith('.safetensors')}
    if {path.name for path in model.glob('*.safetensors')} != expected_weights:
        reasons.append('unexpected_weight_files')
    model_hashes = {}
    for name, expected in MODEL_HASHES.items():
        path = model / name
        model_hashes[name] = file_hash(path) if path.is_file() else None
        if model_hashes[name] != expected:
            reasons.append('model_artifact_mismatch:' + name)
    command = [sys.executable, '-m', 'vllm_apple.mlx_gemma2_compat', '--model', str(model),
               '--host', '127.0.0.1', '--port', str(port), '--decode-concurrency', '1',
               '--prompt-concurrency', '1', '--prefill-step-size', '512', '--prompt-cache-size', '4']
    return dict(schema_version=1, scope='local_text_preview', eligible_for_preview=not reasons,
                rejection_reasons=reasons, packages=packages, hardware=hardware.to_dict(),
                source_sha256=source_hashes, model_sha256=model_hashes, launch_command=command,
                limits=dict(context_tokens=4096, output_tokens=512, allocator_bytes=8*1024**3,
                            cache_bytes=policy['allocator_cache_bytes'], connections=16, concurrency=1),
                efficiency=policy,
                gpu_verified=False, production_qualified=False, performance_qualified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--language', choices=('en', 'ja', 'zh'), default='en')
    parser.add_argument('--efficiency', choices=('baseline', 'responsive', 'compact'), default='baseline',
                        help='Explicit experimental tuning; baseline preserves current settings')
    parser.add_argument('--check', action='store_true', help='Check artifacts without starting GPU work')
    args = parser.parse_args(argv)
    try:
        plan = preview_plan(args.model, args.port, args.efficiency)
    except (ValueError, OSError) as error:
        print(json.dumps(dict(error_code='preview_check_failed', detail=str(error),
                              message=MESSAGES[args.language]), ensure_ascii=False), file=sys.stderr)
        return 1
    plan['message'] = MESSAGES[args.language]
    if args.check or not plan['eligible_for_preview']:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0 if plan['eligible_for_preview'] else 1
    print(MESSAGES[args.language], file=sys.stderr, flush=True)
    environment = dict(os.environ, VLLM_APPLE_P1_PROFILE='1', VLLM_APPLE_P1_EFFICIENCY=args.efficiency)
    os.execve(sys.executable, plan['launch_command'], environment)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
