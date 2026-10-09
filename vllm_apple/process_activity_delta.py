"""Compare valid process counters within one worker epoch; no causal claims."""
COUNTERS = ('faults', 'pageins', 'cow_faults', 'context_switches')


def activity_delta(before, after):
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {'available': False, 'reason': 'missing_sample'}
    if before.get('available') is not True or after.get('available') is not True:
        return {'available': False, 'reason': 'unavailable_sample'}
    pid = before.get('pid')
    if type(pid) is not int or pid <= 0 or after.get('pid') != pid or type(after.get('pid')) is not int:
        return {'available': False, 'reason': 'different_or_invalid_pid'}
    if before.get('source') != 'Darwin PROC_PIDTASKINFO' or after.get('source') != before['source']:
        return {'available': False, 'reason': 'different_or_unknown_source'}
    for key in (*COUNTERS, 'resident_size'):
        if any(type(sample.get(key)) is not int or sample[key] < 0 for sample in (before, after)):
            return {'available': False, 'reason': 'invalid_counter'}
    if any(after[key] < before[key] for key in COUNTERS):
        return {'available': False, 'reason': 'counter_reset_or_overflow'}
    return {'available': True, 'pid': pid, 'counters': {key: after[key]-before[key] for key in COUNTERS},
            'rss_change_bytes': after['resident_size']-before['resident_size'],
            'scope': 'same PID, benchmark window, process-wide; not causal attribution'}


def memory_activity_delta(before, after):
    """Join independently reported RSS/allocator changes; never subtract as categories."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {'available': False, 'reason': 'missing_resources'}
    result = activity_delta(before.get('process_activity'), after.get('process_activity'))
    if result['available'] is not True:
        return result
    allocators = [sample.get('allocator') for sample in (before, after)]
    keys = ('active_bytes', 'cache_bytes')
    if any(not isinstance(sample, dict) or any(type(sample.get(key)) is not int or sample[key] < 0
                                              for key in keys) for sample in allocators):
        result['allocator'] = {'available': False, 'reason': 'missing_or_invalid_allocator'}
    else:
        result['allocator'] = {'available': True,
            'change_bytes': {key: allocators[1][key]-allocators[0][key] for key in keys},
            'scope': 'non-atomic allocator counters; not additive RSS categories'}
    return result
