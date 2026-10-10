"""Operating conditions are admission constraints, not performance predictions."""
from vllm_apple.benchmark_context import observe_benchmark_context


def observe_conditions():
    context = observe_benchmark_context()
    return {key: context[key] for key in ('thermal_state', 'power_source', 'power_mode')}


def require_matching_conditions(measured, current):
    for context in (measured, current):
        if not isinstance(context, dict) or set(context) != {
                'thermal_state', 'power_source', 'power_mode'}:
            raise ValueError('operating conditions missing')
        if (context['thermal_state'] != 'nominal'
                or context['power_source'] not in ('AC Power', 'Battery Power')
                or context['power_mode'] not in ('automatic', 'low_power', 'high_power')):
            raise ValueError('operating conditions unsafe or unknown')
    if measured != current:
        raise ValueError('measured operating conditions changed')
