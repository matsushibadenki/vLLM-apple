import json,os,sys,time
from pathlib import Path
sys.path.insert(0,'/Users/Shared/Program/vLLM-apple')
from vllm_apple.backend import BackendConfig,BackendProcess
from vllm_apple.phase_probe import PhaseProbeConfig
from vllm_apple.text_benchmark import run_text_benchmark
from scripts.qualify_gemma2_batch_mask import _long_prefix_cases,_idle_resources,_cancel_stream,_rss_trend,_file_sha256
root=Path('/Users/Shared/Program/vLLM-apple');out=root/'docs/evaluation/p1-spm-reuse-confirm-m4-2026-10-10';out.mkdir(exist_ok=False)
os.environ.update(PYTHONPATH=str(root),HF_HUB_OFFLINE='1',VLLM_APPLE_P1_PROFILE='1',VLLM_APPLE_P1_EFFICIENCY='compact')
for index,priority in enumerate(('off','on')):
 os.environ['VLLM_APPLE_P1_SPM_REUSE']=priority
 b=BackendProcess(BackendConfig(model=str(root/'models/gemma-2-2b-it-4bit'),executable=Path('/opt/homebrew/opt/vllm-metal/libexec/bin/python'),python_module='vllm_apple.mlx_gemma2_compat',backend_kind='mlx_lm',port=19176,startup_timeout=90,extra_arguments=('--decode-concurrency','2','--prompt-concurrency','2','--prefill-step-size','512','--prompt-cache-size','4')))
 row=dict(spm_reuse=priority,windows=[],resources=[],scope='fixed eight mixed cycles; not long-run qualification',source_sha256=_file_sha256(root/'vllm_apple/mlx_gemma2_compat.py'))
 try:
  b.start();config=PhaseProbeConfig('http://127.0.0.1:19176',str(root/'models/gemma-2-2b-it-4bit'),'Apple-M4-32GiB',maximum_output_tokens=16,target_pid=b.pid,timeout_seconds=30)
  row['warmup']=run_text_benchmark(config,requests=3)
  row['prime_long']=run_text_benchmark(config,requests=3,concurrency=1,cases=_long_prefix_cases(),ttft_slo_ms=10000,e2e_slo_ms=20000)
  start=time.monotonic()
  for cycle in range(8):
   short=run_text_benchmark(config,requests=12,concurrency=2,ttft_slo_ms=5000,e2e_slo_ms=10000)
   cancel=_cancel_stream(19176,str(root/'models/gemma-2-2b-it-4bit'))
   long=run_text_benchmark(config,requests=3,concurrency=1,cases=_long_prefix_cases(),ttft_slo_ms=10000,e2e_slo_ms=20000)
   row['windows'].append(dict(cycle=cycle,short=short,cancel=cancel,long=long))
   res=_idle_resources(19176);res['elapsed_seconds']=time.monotonic()-start;row['resources'].append(res)
  row['rss_trend']=_rss_trend([dict(elapsed_seconds=r['elapsed_seconds'],rss_bytes=r['process_activity']['resident_size'],pid=b.pid) for r in row['resources']])
 finally:
  b.stop();row['stopped']=not b.running
  (out/f'{index}-{priority}.json').write_text(json.dumps(row,indent=2)+'\n')
 print(json.dumps(dict(index=index,spm_reuse=priority,quality=sum(w[k]['quality_passed'] for w in row['windows'] for k in ('short','long')),slo=sum(w[k]['slo_quality_passed'] for w in row['windows'] for k in ('short','long')),requests=120,rss=row.get('rss_trend'))),flush=True)
