"""Bounded native Metal performance qualification for representative operators."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
from pathlib import Path

MAX_NATIVE_BENCHMARK_OUTPUT = 256 * 1024
_OPERATORS = {
    "gpu_gemm", "gpu_gemv", "unified_memory_copy", "metal_launch", "attention",
    "int8_matmul", "packed_u4_unpack", "nf4_gemv_fused", "nf4_decode_gemv_two_pass",
    "nf4_gemm_fused", "nf4_decode_gemm_two_pass", "nf4_attention_fused",
    "nf4_decode_attention_three_pass",
}

_SWIFT_SOURCE = r'''
import Foundation
import Metal

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(message.utf8)); exit(2)
}
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { fail("metal_unavailable") }
let source = """
#include <metal_stdlib>
using namespace metal;
kernel void empty_kernel(device uint *out [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0) out[0] += 1; }
kernel void gemm(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *c [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
 if (p.x >= 64 || p.y >= 64) return; float s = 0; for (uint k=0;k<64;k++) s += a[p.y*64+k]*b[k*64+p.x]; c[p.y*64+p.x]=s;
}
kernel void gemv(device const float *a [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
 if (i >= 256) return; float s=0; for(uint k=0;k<256;k++) s += a[i*256+k]*x[k]; y[i]=s;
}
kernel void copy_shared(device const uint *a [[buffer(0)]], device uint *b [[buffer(1)]], uint i [[thread_position_in_grid]]) { if(i < 8388608) b[i]=a[i]; }
kernel void attention(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]], device float *o [[buffer(3)]], uint i [[thread_position_in_grid]]) {
 if(i>=64) return; float m=-INFINITY; for(uint t=0;t<128;t++){float s=0;for(uint d=0;d<64;d++)s+=q[d]*k[t*64+d];m=max(m,s*0.125f);} float z=0,r=0; for(uint t=0;t<128;t++){float s=0;for(uint d=0;d<64;d++)s+=q[d]*k[t*64+d];float w=exp(s*0.125f-m);z+=w;r+=w*v[t*64+i];} o[i]=r/z;
}
kernel void int8_matmul(device const char *a [[buffer(0)]], device const char *b [[buffer(1)]], device int *c [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
 if(p.x>=64||p.y>=64)return; int s=0;for(uint k=0;k<64;k++)s+=int(a[p.y*64+k])*int(b[k*64+p.x]);c[p.y*64+p.x]=s;
}
kernel void packed_u4_unpack(device const uchar *packed [[buffer(0)]], device uint *codes [[buffer(1)]], uint i [[thread_position_in_grid]]) {
 if(i>=65536)return; uchar value=packed[i>>1];codes[i]=(i&1)?uint(value>>4):uint(value&15);
}
inline float nf4_value(uint code) {
 switch(code){case 0:return -1.0f;case 1:return -0.6961928009986877f;case 2:return -0.5250730514526367f;case 3:return -0.39491748809814453f;case 4:return -0.28444138169288635f;case 5:return -0.18477343022823334f;case 6:return -0.09105003625154495f;case 7:return 0.0f;case 8:return 0.07958029955625534f;case 9:return 0.16093020141124725f;case 10:return 0.24611230194568634f;case 11:return 0.33791524171829224f;case 12:return 0.44070982933044434f;case 13:return 0.5626170039176941f;case 14:return 0.7229568362236023f;default:return 1.0f;}
}
kernel void nf4_gemv_fused(device const uchar *packed [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], uint row [[thread_position_in_grid]]) {
 if(row>=256)return;float sum=0;uint base=row*256;for(uint column=0;column<256;column++){uint i=base+column;uchar value=packed[i>>1];uint code=(i&1)?uint(value>>4):uint(value&15);sum+=nf4_value(code)*x[column];}y[row]=sum;
}
kernel void nf4_decode(device const uchar *packed [[buffer(0)]], device float *weights [[buffer(1)]], uint i [[thread_position_in_grid]]) {
 if(i>=65536)return;uchar value=packed[i>>1];uint code=(i&1)?uint(value>>4):uint(value&15);weights[i]=nf4_value(code);
}
kernel void nf4_gemm_fused(device const uchar *packed [[buffer(0)]], device const float *b [[buffer(1)]], device float *c [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
 if(p.x>=64||p.y>=64)return;float s=0;for(uint k=0;k<64;k++){uint i=p.y*64+k;uchar value=packed[i>>1];uint code=(i&1)?uint(value>>4):uint(value&15);s+=nf4_value(code)*b[k*64+p.x];}c[p.y*64+p.x]=s;
}
kernel void nf4_attention_fused(device const float *q [[buffer(0)]], device const uchar *packedK [[buffer(1)]], device const uchar *packedV [[buffer(2)]], device float *o [[buffer(3)]], uint i [[thread_position_in_grid]]) {
 if(i>=64)return;float m=-INFINITY;for(uint t=0;t<128;t++){float s=0;for(uint d=0;d<64;d++){uint j=t*64+d;uchar p=packedK[j>>1];s+=q[d]*nf4_value((j&1)?uint(p>>4):uint(p&15));}m=max(m,s*0.125f);}float z=0,r=0;for(uint t=0;t<128;t++){float s=0;for(uint d=0;d<64;d++){uint j=t*64+d;uchar p=packedK[j>>1];s+=q[d]*nf4_value((j&1)?uint(p>>4):uint(p&15));}float w=exp(s*0.125f-m);uint j=t*64+i;uchar p=packedV[j>>1];z+=w;r+=w*nf4_value((j&1)?uint(p>>4):uint(p&15));}o[i]=r/z;
}
"""
let library = try! device.makeLibrary(source: source, options: nil)
func pipeline(_ name: String) -> MTLComputePipelineState { try! device.makeComputePipelineState(function: library.makeFunction(name: name)!) }
func buffer<T>(_ values: [T]) -> MTLBuffer { device.makeBuffer(bytes: values, length: values.count * MemoryLayout<T>.stride, options: .storageModeShared)! }
func measure(_ pipeline: MTLComputePipelineState, _ buffers: [MTLBuffer], _ grid: MTLSize, _ group: MTLSize, _ samples: Int) -> [UInt64] {
 func once() -> UInt64 { let c=queue.makeCommandBuffer()!;let e=c.makeComputeCommandEncoder()!;e.setComputePipelineState(pipeline);for(i,b) in buffers.enumerated(){e.setBuffer(b,offset:0,index:i)};let s=DispatchTime.now().uptimeNanoseconds;e.dispatchThreads(grid,threadsPerThreadgroup:group);e.endEncoding();c.commit();c.waitUntilCompleted();if c.status != .completed { fail("command_failed") };return max(1,DispatchTime.now().uptimeNanoseconds-s) }
 _ = once(); return (0..<samples).map { _ in once() }
}
func measureTwo(_ first: MTLComputePipelineState, _ firstBuffers:[MTLBuffer], _ firstGrid:MTLSize, _ firstGroup:MTLSize, _ second:MTLComputePipelineState, _ secondBuffers:[MTLBuffer], _ secondGrid:MTLSize, _ secondGroup:MTLSize, _ samples:Int) -> [UInt64] {
 func once()->UInt64{let c=queue.makeCommandBuffer()!;let e=c.makeComputeCommandEncoder()!;let s=DispatchTime.now().uptimeNanoseconds;e.setComputePipelineState(first);for(i,b) in firstBuffers.enumerated(){e.setBuffer(b,offset:0,index:i)};e.dispatchThreads(firstGrid,threadsPerThreadgroup:firstGroup);e.memoryBarrier(scope:.buffers);e.setComputePipelineState(second);for(i,b) in secondBuffers.enumerated(){e.setBuffer(b,offset:0,index:i)};e.dispatchThreads(secondGrid,threadsPerThreadgroup:secondGroup);e.endEncoding();c.commit();c.waitUntilCompleted();if c.status != .completed { fail("command_failed") };return max(1,DispatchTime.now().uptimeNanoseconds-s)}
 _=once();return(0..<samples).map{_ in once()}
}
func measureThree(_ first:MTLComputePipelineState,_ firstBuffers:[MTLBuffer],_ firstGrid:MTLSize,_ firstGroup:MTLSize,_ second:MTLComputePipelineState,_ secondBuffers:[MTLBuffer],_ secondGrid:MTLSize,_ secondGroup:MTLSize,_ third:MTLComputePipelineState,_ thirdBuffers:[MTLBuffer],_ thirdGrid:MTLSize,_ thirdGroup:MTLSize,_ samples:Int)->[UInt64]{
 func once()->UInt64{let c=queue.makeCommandBuffer()!;let e=c.makeComputeCommandEncoder()!;let s=DispatchTime.now().uptimeNanoseconds;for (p,bs,g,t) in [(first,firstBuffers,firstGrid,firstGroup),(second,secondBuffers,secondGrid,secondGroup),(third,thirdBuffers,thirdGrid,thirdGroup)]{e.setComputePipelineState(p);for(i,b) in bs.enumerated(){e.setBuffer(b,offset:0,index:i)};e.dispatchThreads(g,threadsPerThreadgroup:t);e.memoryBarrier(scope:.buffers)};e.endEncoding();c.commit();c.waitUntilCompleted();if c.status != .completed { fail("command_failed") };return max(1,DispatchTime.now().uptimeNanoseconds-s)}
 _=once();return(0..<samples).map{_ in once()}
}
let samples = 7
var reports: [[String: Any]] = []
let one = buffer([UInt32](repeating: 0, count: 1))
reports.append(["operator":"metal_launch","nanoseconds":measure(pipeline("empty_kernel"),[one],MTLSize(width:1,height:1,depth:1),MTLSize(width:1,height:1,depth:1),samples),"work_items":1,"bytes":0,"digest":"launch-\(one.contents().load(as: UInt32.self))"])
let fa = buffer((0..<4096).map { Float(($0 % 13)-6)/13 }); let fb = buffer((0..<4096).map { Float(($0 % 11)-5)/11 }); let fc=device.makeBuffer(length:4096*4,options:.storageModeShared)!
reports.append(["operator":"gpu_gemm","nanoseconds":measure(pipeline("gemm"),[fa,fb,fc],MTLSize(width:64,height:64,depth:1),MTLSize(width:8,height:8,depth:1),samples),"work_items":262144,"bytes":49152,"digest":"\(fc.contents().bindMemory(to:Float.self,capacity:4096)[0])"])
let ma=buffer((0..<65536).map{Float(($0%17)-8)/17});let vx=buffer((0..<256).map{Float(($0%7)-3)/7});let vy=device.makeBuffer(length:1024,options:.storageModeShared)!
reports.append(["operator":"gpu_gemv","nanoseconds":measure(pipeline("gemv"),[ma,vx,vy],MTLSize(width:256,height:1,depth:1),MTLSize(width:64,height:1,depth:1),samples),"work_items":65536,"bytes":263168,"digest":"\(vy.contents().bindMemory(to:Float.self,capacity:256)[0])"])
let copyA=buffer([UInt32](repeating:0x5a5a5a5a,count:8388608));let copyB=device.makeBuffer(length:33554432,options:.storageModeShared)!
reports.append(["operator":"unified_memory_copy","nanoseconds":measure(pipeline("copy_shared"),[copyA,copyB],MTLSize(width:8388608,height:1,depth:1),MTLSize(width:256,height:1,depth:1),samples),"work_items":8388608,"bytes":67108864,"digest":"\(copyB.contents().load(as:UInt32.self))"])
let q=buffer((0..<64).map{Float(($0%5)-2)/5});let ak=buffer((0..<8192).map{Float(($0%9)-4)/9});let av=buffer((0..<8192).map{Float(($0%7)-3)/7});let ao=device.makeBuffer(length:256,options:.storageModeShared)!
reports.append(["operator":"attention","nanoseconds":measure(pipeline("attention"),[q,ak,av,ao],MTLSize(width:64,height:1,depth:1),MTLSize(width:64,height:1,depth:1),samples),"work_items":1048576,"bytes":66048,"digest":"\(ao.contents().bindMemory(to:Float.self,capacity:64)[0])"])
let ia=buffer((0..<4096).map{Int8(($0%15)-7)});let ib=buffer((0..<4096).map{Int8(($0%13)-6)});let ic=device.makeBuffer(length:16384,options:.storageModeShared)!
reports.append(["operator":"int8_matmul","nanoseconds":measure(pipeline("int8_matmul"),[ia,ib,ic],MTLSize(width:64,height:64,depth:1),MTLSize(width:8,height:8,depth:1),samples),"work_items":262144,"bytes":24576,"digest":"\(ic.contents().bindMemory(to:Int32.self,capacity:4096)[0])"])
let packedValues:[UInt8]=(0..<32768).map{index in let low=UInt8((index*2)&15);let high=UInt8((index*2+1)&15);return low|(high<<4)}
let packed=buffer(packedValues);let unpacked=device.makeBuffer(length:65536*4,options:.storageModeShared)!
reports.append(["operator":"packed_u4_unpack","nanoseconds":measure(pipeline("packed_u4_unpack"),[packed,unpacked],MTLSize(width:65536,height:1,depth:1),MTLSize(width:256,height:1,depth:1),samples),"work_items":65536,"bytes":294912,"digest":"u4-\(unpacked.contents().bindMemory(to:UInt32.self,capacity:65536)[65535])"])
for i in 0..<65536 { if unpacked.contents().bindMemory(to:UInt32.self,capacity:65536)[i] != UInt32(i&15) { fail("u4_correctness_failed") } }
let nx=buffer([Float](repeating:1,count:256));let ny=device.makeBuffer(length:256*4,options:.storageModeShared)!
reports.append(["operator":"nf4_gemv_fused","nanoseconds":measure(pipeline("nf4_gemv_fused"),[packed,nx,ny],MTLSize(width:256,height:1,depth:1),MTLSize(width:64,height:1,depth:1),samples),"work_items":65536,"bytes":34816,"digest":"nf4-\(ny.contents().bindMemory(to:Float.self,capacity:256)[0])"])
let decoded=device.makeBuffer(length:65536*4,options:.storageModeShared)!;let twoPassY=device.makeBuffer(length:256*4,options:.storageModeShared)!
reports.append(["operator":"nf4_decode_gemv_two_pass","nanoseconds":measureTwo(pipeline("nf4_decode"),[packed,decoded],MTLSize(width:65536,height:1,depth:1),MTLSize(width:256,height:1,depth:1),pipeline("gemv"),[decoded,nx,twoPassY],MTLSize(width:256,height:1,depth:1),MTLSize(width:64,height:1,depth:1),samples),"work_items":65536,"bytes":559104,"digest":"nf4-two-\(twoPassY.contents().bindMemory(to:Float.self,capacity:256)[0])"])
let table:[Float]=[-1,-0.6961928,-0.52507305,-0.3949175,-0.28444138,-0.18477343,-0.09105004,0,0.0795803,0.1609302,0.2461123,0.33791524,0.44070983,0.562617,0.72295684,1]
var expected:Float=0;for i in 0..<256 { expected += table[i&15] };if abs(ny.contents().bindMemory(to:Float.self,capacity:256)[0]-expected)>0.0001 { fail("nf4_gemv_correctness_failed") }
if abs(twoPassY.contents().bindMemory(to:Float.self,capacity:256)[0]-expected)>0.0001 { fail("nf4_two_pass_correctness_failed") }
let nf4Gemm=device.makeBuffer(length:4096*4,options:.storageModeShared)!;let decodedGemm=device.makeBuffer(length:4096*4,options:.storageModeShared)!;let twoPassGemm=device.makeBuffer(length:4096*4,options:.storageModeShared)!
reports.append(["operator":"nf4_gemm_fused","nanoseconds":measure(pipeline("nf4_gemm_fused"),[packed,fb,nf4Gemm],MTLSize(width:64,height:64,depth:1),MTLSize(width:8,height:8,depth:1),samples),"work_items":262144,"bytes":26624,"digest":"nf4-gemm-\(nf4Gemm.contents().bindMemory(to:Float.self,capacity:4096)[0])"])
reports.append(["operator":"nf4_decode_gemm_two_pass","nanoseconds":measureTwo(pipeline("nf4_decode"),[packed,decodedGemm],MTLSize(width:4096,height:1,depth:1),MTLSize(width:256,height:1,depth:1),pipeline("gemm"),[decodedGemm,fb,twoPassGemm],MTLSize(width:64,height:64,depth:1),MTLSize(width:8,height:8,depth:1),samples),"work_items":262144,"bytes":59392,"digest":"nf4-gemm-two-\(twoPassGemm.contents().bindMemory(to:Float.self,capacity:4096)[0])"])
if abs(nf4Gemm.contents().bindMemory(to:Float.self,capacity:4096)[0]-twoPassGemm.contents().bindMemory(to:Float.self,capacity:4096)[0])>0.0001 { fail("nf4_gemm_correctness_failed") }
let packedAttention=buffer((0..<4096).map{index in let low=UInt8((index*2)&15);let high=UInt8((index*2+1)&15);return low|(high<<4)});let nf4Attention=device.makeBuffer(length:256,options:.storageModeShared)!;let decodedK=device.makeBuffer(length:8192*4,options:.storageModeShared)!;let decodedV=device.makeBuffer(length:8192*4,options:.storageModeShared)!;let decodedAttention=device.makeBuffer(length:256,options:.storageModeShared)!
reports.append(["operator":"nf4_attention_fused","nanoseconds":measure(pipeline("nf4_attention_fused"),[q,packedAttention,packedAttention,nf4Attention],MTLSize(width:64,height:1,depth:1),MTLSize(width:64,height:1,depth:1),samples),"work_items":1048576,"bytes":33024,"digest":"nf4-attention-\(nf4Attention.contents().bindMemory(to:Float.self,capacity:64)[0])"])
reports.append(["operator":"nf4_decode_attention_three_pass","nanoseconds":measureThree(pipeline("nf4_decode"),[packedAttention,decodedK],MTLSize(width:8192,height:1,depth:1),MTLSize(width:256,height:1,depth:1),pipeline("nf4_decode"),[packedAttention,decodedV],MTLSize(width:8192,height:1,depth:1),MTLSize(width:256,height:1,depth:1),pipeline("attention"),[q,decodedK,decodedV,decodedAttention],MTLSize(width:64,height:1,depth:1),MTLSize(width:64,height:1,depth:1),samples),"work_items":1048576,"bytes":131328,"digest":"nf4-attention-three-\(decodedAttention.contents().bindMemory(to:Float.self,capacity:64)[0])"])
for i in 0..<64 { if abs(nf4Attention.contents().bindMemory(to:Float.self,capacity:64)[i]-decodedAttention.contents().bindMemory(to:Float.self,capacity:64)[i])>0.0001 { fail("nf4_attention_correctness_failed") } }
let output:[String:Any]=["device":device.name,"samples":samples,"reports":reports]
let data=try! JSONSerialization.data(withJSONObject:output,options:[.sortedKeys]);print(String(data:data,encoding:.utf8)!)
'''


def run_native_hardware_benchmarks(*, timeout_seconds: float = 120) -> dict[str, object]:
    if not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 600:
        raise ValueError("invalid native benchmark timeout")
    completed = subprocess.run(
        ["/usr/bin/swift", "-"], input=_SWIFT_SOURCE, text=True,
        capture_output=True, timeout=timeout_seconds, check=False,
    )
    if completed.returncode != 0 or len(completed.stdout.encode()) > MAX_NATIVE_BENCHMARK_OUTPUT:
        raise RuntimeError("native Metal benchmark failed")
    payload = json.loads(completed.stdout)
    if (not isinstance(payload, dict) or set(payload) != {"device", "samples", "reports"}
            or not isinstance(payload["device"], str) or payload["samples"] != 7
            or not isinstance(payload["reports"], list) or len(payload["reports"]) != len(_OPERATORS)):
        raise ValueError("invalid native benchmark response")
    normalized = []
    identities = set()
    for report in payload["reports"]:
        if not isinstance(report, dict) or set(report) != {"operator", "nanoseconds", "work_items", "bytes", "digest"}:
            raise ValueError("invalid native benchmark measurement")
        operator = report["operator"]
        values = report["nanoseconds"]
        if (operator not in _OPERATORS or operator in identities or not isinstance(values, list)
                or len(values) != 7 or any(type(value) is not int or value <= 0 for value in values)
                or type(report["work_items"]) is not int or report["work_items"] <= 0
                or type(report["bytes"]) is not int or report["bytes"] < 0
                or not isinstance(report["digest"], str) or not report["digest"]):
            raise ValueError("invalid native benchmark measurement")
        identities.add(operator)
        ordered = sorted(values)
        median = ordered[len(ordered) // 2]
        normalized.append({
            "operator": operator,
            "nanoseconds": values,
            "median_nanoseconds": median,
            "work_items": report["work_items"],
            "throughput_work_items_per_second": report["work_items"] * 1_000_000_000 / median,
            "bytes": report["bytes"],
            "bandwidth_bytes_per_second": (
                report["bytes"] * 1_000_000_000 / median if report["bytes"] else None
            ),
            "output_digest": hashlib.sha256(report["digest"].encode()).hexdigest(),
        })
    if identities != _OPERATORS:
        raise ValueError("native benchmark response is incomplete")
    result = {
        "schema_version": 1,
        "passed": True,
        "device": payload["device"],
        "sample_count": 7,
        "reports": sorted(normalized, key=lambda value: value["operator"]),
    }
    result["report_id"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def save_native_hardware_benchmarks(report: dict[str, object], path: Path) -> Path:
    encoded = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    if not 1 <= len(encoded) <= MAX_NATIVE_BENCHMARK_OUTPUT:
        raise ValueError("native benchmark report exceeds bound")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = path.parent.lstat()
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid()
            or stat.S_IMODE(parent.st_mode) & 0o077):
        raise ValueError("native benchmark report directory must be private")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path
