"""Collect and export one bounded Metal GPU trace of the fixed real-model workload."""
from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from .trace import add_shader_breakdown, sha, summarize_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True,exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    trace = output/'workload.trace'
    workload = output/'workload.json'
    command = ['xcrun','xctrace','record','--template','Metal System Trace',
        '--instrument','Metal GPU Counters','--time-limit','30s','--no-prompt',
        '--output',str(trace),'--env',f'PYTHONPATH={root}','--env','HF_HUB_OFFLINE=1',
        '--target-stdout',str(output/'worker.log'),'--launch','--',str(args.python.absolute()),
        '-m','experiments.p3_mlx.workload','--model',str(args.model.resolve()),'--output',str(workload)]
    with (output/'record.log').open('w') as log:
        subprocess.run(command,cwd=root,stdout=log,stderr=log,check=True,timeout=120)
    toc = output/'toc.xml'
    subprocess.run(['xcrun','xctrace','export','--input',str(trace),'--toc','--output',str(toc)],check=True,timeout=45)
    launch = ET.parse(toc).find('.//target/process')
    if launch is None:
        launch = ET.parse(toc).find('.//process[@type="launched"]')
    if launch is None or launch.attrib.get('return-exit-status') != '0':
        raise ValueError('profile workload did not exit cleanly')
    pid = int(launch.attrib['pid'])
    for schema in ('metal-gpu-intervals','time-info','metal-shader-profiler-intervals'):
        subprocess.run(['xcrun','xctrace','export','--input',str(trace),'--xpath',
            f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
            '--output',str(output/f'{schema}.xml')],check=True,timeout=45)
    metadata = json.loads(workload.read_text())
    if not metadata['identity_unchanged']:
        raise ValueError('profile identity changed')
    report = summarize_trace(output/'metal-gpu-intervals.xml',output/'time-info.xml',workload,pid)
    add_shader_breakdown(report,output/'metal-shader-profiler-intervals.xml',output/'time-info.xml',workload,pid)
    report.update(identity=metadata['identity'],identity_unchanged=True,
        operating_conditions=metadata['operating_conditions'],
        command=command,collector_source_sha256=sha(Path(__file__)),
        trace_path=str(trace),trace_files={str(p.relative_to(trace)):sha(p) for p in sorted(trace.rglob('*')) if p.is_file()},
        trace_sha256_scope='per-file inventory; local diagnostic trace is retained outside versioned evidence',
        profiler_overhead='timings are instrumented; E2E selection uses separate unprofiled HTTP workers')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['shader_time_breakdown'],indent=2))


if __name__ == '__main__':
    main()
