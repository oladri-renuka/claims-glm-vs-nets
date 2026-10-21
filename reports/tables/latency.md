CPU: Apple M4, 1 torch thread, 1000 sequential requests. Bundle size: 0.14 MB.

| path | p50 (ms) | p95 (ms) |
|---|---|---|
| inprocess_nn | 12.65 | 13.16 |
| inprocess_glm | 4.74 | 5.02 |
| http_nn | 13.95 | 14.67 |
| http_glm | 5.55 | 6.27 |
