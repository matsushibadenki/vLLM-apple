"""Optional, bounded backend observations; never infer KV hits from capacity."""
import json
import urllib.error
import urllib.request

from .phase_probe import PhaseProbeConfig


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def observe_backend_memory(config: PhaseProbeConfig) -> dict[str, object]:
    request = urllib.request.Request(config.base_url.rstrip('/') + '/v1/vllm-apple/memory')
    if config.session_token:
        request.add_header('Authorization', 'Bearer ' + config.session_token)
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=1) as response:
            raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError('oversized')
        payload = json.loads(raw)
        fields = ('active_bytes', 'cache_bytes', 'peak_bytes', 'kv_cache_bytes')
        if (not isinstance(payload, dict) or type(payload.get('schema_version')) is not int
                or payload['schema_version'] != 1):
            raise ValueError('schema')
        if any(type(payload.get(key)) is not int or payload[key] < 0 for key in fields):
            raise ValueError('counter')
        if type(payload.get('traversal_complete')) is not bool:
            raise ValueError('completeness')
        tokens = payload.get('kv_cache_tokens')
        if tokens is not None and (type(tokens) is not int or tokens < 0):
            raise ValueError('tokens')
        source = payload.get('kv_measurement_source', 'unspecified')
        if source not in ('unspecified', 'bounded_array_traversal', 'backend_lru_accounting'):
            raise ValueError('source')
        consistency = payload.get('snapshot_consistency', 'unspecified')
        if consistency not in ('unspecified', 'non_atomic'):
            raise ValueError('consistency')
        return {'status': 'observed', 'memory': {key: payload[key] for key in fields},
                'kv_cache_tokens': tokens,
                'kv_measurement_source': source,
                'snapshot_consistency': consistency,
                'traversal_complete': payload['traversal_complete'],
                'cache_hits': None, 'cache_evictions': None}
    except urllib.error.HTTPError as error:
        return {'status': 'unavailable', 'reason': 'http_error', 'http_status': error.code}
    except (OSError, ValueError):
        return {'status': 'unavailable', 'reason': 'transport_or_invalid_payload'}
