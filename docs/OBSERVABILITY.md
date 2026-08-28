# Request observability

The runtime correlates one request across the HTTP or Unix-domain-socket API,
the local vLLM backend proxy, SSE handling, and request-scoped runtime events.

## Request IDs

- Clients may send `X-Request-ID` using 8–64 ASCII letters, digits, `.`, `_`, or `-`.
- Missing or invalid values are replaced with an opaque UUID.
- Every JSON and SSE response returns the resolved ID.
- The backend proxy forwards the same ID without adding it to the JSON body.
- Events published in that request scope include `payload.request_id`.

## Structured request log

Each server owns an in-memory, thread-safe ring containing at most 256 records.
Records contain only request ID, method, normalized route, status, duration,
streaming state, and an optional error code. Query strings, authorization data,
prompts, messages, generated text, and request or response bodies are never stored.
The bounded ring prevents observability data from growing daemon memory over time.
