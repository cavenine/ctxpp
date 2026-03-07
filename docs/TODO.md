# TODO

- Re-evaluate `github.com/coder/hnsw` after the ANN spike. If maintenance, API stability, persistence limits, or delete/update behavior become blockers, consider forking it and carrying ctx++-specific fixes or lifecycle improvements.
- If a fork becomes necessary, document the exact reasons first: missing features, bug fixes, performance tuning, persistence changes, or maintenance risk.
- Prefer staying on upstream if the integration remains simple and the library behaves well under ctx++ benchmarks and incremental indexing workloads.
- If we need a pure-Go alternative to `github.com/coder/hnsw`, the first spike candidate should be `github.com/wizenheimer/comet`; it appears to support persistence plus insert/delete without adding CGO or a separate service.
