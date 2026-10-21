CPU: Apple M4, 1 torch thread, 1000 sequential requests. Bundle size: 0.15 MB.

| path | p50 (ms) | p95 (ms) |
|---|---|---|
| inprocess_nn | 8.10 | 8.42 |
| inprocess_glm | 6.09 | 6.39 |
| http_nn | 9.20 | 9.80 |
| http_glm | 7.14 | 7.73 |
