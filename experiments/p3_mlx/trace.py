"""Bounded Instruments XML decoder; intersect GPU intervals with workload windows."""
from __future__ import annotations

import hashlib
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

MAX_XML_BYTES = 128*1024**2


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def table(path):
    path = Path(path)
    if path.stat().st_size > MAX_XML_BYTES:
        raise ValueError('trace XML exceeds 128 MiB')
    raw = path.read_bytes()
    if b'<!DOCTYPE' in raw or b'<!ENTITY' in raw:
        raise ValueError('DTD/entity trace XML is unsupported')
    root = ET.fromstring(raw)
    nodes = root.findall('node')
    if len(nodes) != 1:
        raise ValueError('export exactly one trace table')
    schema = nodes[0].find('schema')
    fields = [c.findtext('mnemonic') for c in schema.findall('col')]
    ids = {}
    for element in root.iter():
        if 'id' in element.attrib:
            if element.attrib['id'] in ids:
                raise ValueError('duplicate trace ID')
            ids[element.attrib['id']] = element

    def resolve(element):
        seen = set()
        while 'ref' in element.attrib:
            key = element.attrib['ref']
            if key in seen or key not in ids:
                raise ValueError('invalid trace reference')
            seen.add(key)
            element = ids[key]
        return element

    rows = nodes[0].findall('row')
    if len(rows) > 100_000:
        raise ValueError('trace row budget exceeded')
    values = []
    for row in rows:
        if len(row) != len(fields):
            raise ValueError('trace column count mismatch')
        values.append({k: resolve(v) for k, v in zip(fields, row)})
    return values, resolve


def union_ns(intervals):
    total = 0
    end = None
    for left, right in sorted(intervals):
        if right <= left:
            raise ValueError('nonpositive trace interval')
        total += right-max(left, end if end is not None else left) if end is None or right > end else 0
        end = max(right, end if end is not None else right)
    return total


def summarize_trace(gpu_xml, time_xml, workload_json, pid):
    clocks, resolve_clock = table(time_xml)
    if len(clocks) != 1:
        raise ValueError('unsupported multi-epoch clock mapping')
    clock = clocks[0]
    info = list(clock['timebase-info'])
    numer, denom = [int(resolve_clock(v).text) for v in info]
    if numer <= 0 or denom <= 0:
        raise ValueError('invalid trace clock')
    origin = int(clock['mabs-epoch'].text)*numer//denom
    rows, resolve = table(gpu_xml)
    intervals = []
    latency = []
    for row in rows:
        process = row['process']
        node = process.find('pid')
        if node is None or int(resolve(node).text) != pid:
            continue
        if row['channel-name'].text != 'Compute' or row['state'].text != 'Active' or row['event-depth'].text != '0':
            continue
        start = int(row['start'].text)
        duration = int(row['duration'].text)
        if start < 0 or duration <= 0:
            raise ValueError('invalid GPU interval')
        intervals.append((start, start+duration))
        latency.append((start, int(row['start-latency'].text)))
    if not intervals:
        raise ValueError('no target GPU compute intervals')
    workload = json.loads(Path(workload_json).read_text())
    samples = workload['samples']
    if not 1 <= len(samples) <= 1000:
        raise ValueError('workload sample budget exceeded')
    summaries = {}
    previous_end = 0
    for sample in samples:
        left = sample['start_monotonic_ns']-origin
        right = sample['end_monotonic_ns']-origin
        if left < previous_end or right <= left or right-left != sample['wall_ns'] or sample['finite'] is not True:
            raise ValueError('invalid workload window')
        previous_end = right
        clipped = [(max(left, a), min(right, b)) for a, b in intervals if a < right and b > left]
        if not clipped:
            raise ValueError('workload window has no target GPU observation')
        summaries.setdefault(sample['phase'], []).append(dict(
            wall_ns=sample['wall_ns'], gpu_active_ns=union_ns(clipped),
            graph_encode_ns=sample['graph_encode_ns'], cpu_ns=sample['cpu_ns'],
            evaluation_and_wait_ns=sample['evaluation_and_wait_ns'],
            gpu_intervals=len(clipped),
            submission_latency_ns=[value for a, value in latency if left <= a < right]))
    result = {}
    for phase, group in summaries.items():
        result[phase] = {k:statistics.median(row[k] for row in group)
            for k in ('wall_ns', 'gpu_active_ns', 'graph_encode_ns', 'cpu_ns', 'evaluation_and_wait_ns', 'gpu_intervals')}
        result[phase]['samples'] = len(group)
        result[phase]['gpu_active_wall_fraction'] = sum(r['gpu_active_ns'] for r in group)/sum(r['wall_ns'] for r in group)
        waits = [n for row in group for n in row['submission_latency_ns']]
        result[phase]['median_submission_latency_ns'] = statistics.median(waits) if waits else None
    return dict(report_kind='p3_gpu_timeline', target_pid=pid, phases=result,
        sources={str(p):sha(p) for p in (gpu_xml, time_xml, workload_json)},
        clock_mapping='mach_absolute_time * numerator / denominator; Python macOS perf_counter monotonic window',
        gpu_time_scope='union of target-PID depth-zero Active Compute intervals; nested/unrelated work excluded',
        bottleneck='gpu_execution' if all(p['gpu_active_wall_fraction'] > .5 for p in result.values()) else 'mixed_or_unresolved',
        bandwidth_or_compute_bound=None, shader_time_breakdown=None,
        qualification=False, inference_speedup_measured=False)


def add_shader_breakdown(report, shader_xml, time_xml, workload_json, pid):
    clocks, resolve_clock = table(time_xml)
    clock = clocks[0]
    numer, denom = [int(resolve_clock(v).text) for v in list(clock['timebase-info'])]
    origin = int(clock['mabs-epoch'].text)*numer//denom
    windows = json.loads(Path(workload_json).read_text())['samples']
    rows, resolve = table(shader_xml)
    totals = {w['phase']: {} for w in windows}

    def category(name):
        name = name.lower()
        if 'qmm' in name or 'qmv' in name or 'quantized' in name:
            return 'quantized_matmul_fused_dequantization'
        if 'attention' in name or 'sdpa' in name:
            return 'attention'
        if 'gemm' in name or 'gemv' in name or 'steel' in name:
            return 'matmul'
        if 'rms' in name or 'norm' in name:
            return 'normalization'
        if 'copy' in name or 'gather' in name or 'scatter' in name:
            return 'copy_or_indexing'
        return 'other'

    for row in rows:
        node = row['process'].find('pid')
        if node is None or int(resolve(node).text) != pid or row['shader-type'].text != 'Compute':
            continue
        left = int(row['start'].text)
        duration = int(row['duration'].text)
        if left < 0 or duration <= 0:
            raise ValueError('invalid shader interval')
        name = row['name'].text or ''
        key = category(name)
        for window in windows:
            a = window['start_monotonic_ns']-origin
            b = window['end_monotonic_ns']-origin
            clipped = min(b,left+duration)-max(a,left)
            if clipped > 0:
                bucket = totals[window['phase']].setdefault(key,dict(sampled_duration_ns=0, shader_intervals=0))
                bucket['sampled_duration_ns'] += clipped
                bucket['shader_intervals'] += 1
    if any(not groups for groups in totals.values()):
        raise ValueError('missing per-phase shader timeline')
    for groups in totals.values():
        total = sum(b['sampled_duration_ns'] for b in groups.values())
        for bucket in groups.values():
            bucket['sampled_time_share'] = bucket['sampled_duration_ns']/total
    report['shader_time_breakdown'] = totals
    report['shader_time_scope'] = 'Instruments Shader Timeline sampled durations, intersected with target-PID workload; shares use summed sampled time, not GPU wall time; fused dequantization cannot be separated'
    report['sources'][str(shader_xml)] = sha(shader_xml)
    report['suggested_tuning_dimension'] = ('prefill_step_size'
        if totals.get('prefill', {}).get('quantized_matmul_fused_dequantization', {}).get('sampled_time_share', 0) > .5
        else None)
    return report
