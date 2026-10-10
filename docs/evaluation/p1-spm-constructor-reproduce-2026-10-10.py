import json,time,tracemalloc,statistics,hashlib
from pathlib import Path
from mlx_lm.utils import load_tokenizer
import mlx_lm.tokenizer_utils as native
from vllm_apple.spm_tokenmap_reuse import install_spm_tokenmap_reuse
model=Path('models/gemma-2-2b-it-4bit');tokenizer=load_tokenizer(model)
reference=tokenizer.detokenizer
assert isinstance(reference,native.SPMStreamingDetokenizer)
def measure():
 samples=[];cpu=[]
 for _ in range(15):
  start=time.perf_counter_ns();c=time.process_time_ns();d=tokenizer.detokenizer
  samples.append(time.perf_counter_ns()-start);cpu.append(time.process_time_ns()-c)
 tracemalloc.start();d=tokenizer.detokenizer;current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
 return dict(instances=15,wall_median_ns=statistics.median(samples),cpu_median_ns=statistics.median(cpu),traced_instance_live_bytes=current,traced_peak_bytes=peak),d
before,_=measure();reuse=install_spm_tokenmap_reuse(native);warm=tokenizer.detokenizer;after,actual=measure()
assert tuple(reference.tokenmap)==actual.tokenmap
cases=[[2,234,256,123,107],[i for i,b in enumerate(reference.tokenmap) if b in (b'\xe4',b'\xb8',b'\xad',b'\xe2\x96\x81')][:8]]
for tokens in cases:
 reference.reset();actual.reset()
 for t in tokens:
  reference.add_token(t);actual.add_token(t);assert reference.text==actual.text
 reference.finalize();actual.finalize();assert reference.text==actual.text
report=dict(scope='warmed SPM constructor only; all vocabulary entries exact; not inference latency/RSS qualification',entries=len(actual.tokenmap),before=before,after=after,reuse=reuse.snapshot(),all_vocabulary_entries_equal=True,incremental_text_equal=True,source_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest())
Path('docs/evaluation/p1-spm-constructor-m4-2026-10-10.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
