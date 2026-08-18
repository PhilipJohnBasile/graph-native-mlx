# Validation Record — v0.2.0

## Portable validation environment

- Linux x86_64
- Python 3.13.5
- MLX and MLX-LM intentionally absent

## Completed checks

- 41 portable tests passed
- all source and test modules compiled with `compileall`
- default YAML graph validated as 11 nodes, 16 edges, and 2 terminals
- generated graph module reproduced byte-for-byte from the YAML source
- schema hash verified as `1b3da7aa892798987fad830163c05aedcf17494f08a1ee483b576facb3ac909a`
- mock fast, deep, repair-success, and repair-abort paths passed
- graph-versus-loop benchmark completed
- direct-provider behavior validated with an injected MLX-LM-compatible backend
- MLX-LM 0.31.3 and newer loader signatures covered by tests
- chat-template failure fallback covered
- final-complete-JSON extraction covered
- controller masks and invalid-transition rejection covered
- policy trace export and dataset validation covered
- controller/provider resume identity guards covered
- wheel installation and source-archive revalidation are performed during release packaging

## Hardware validation boundary

This environment cannot execute Apple Metal or load the selected 27B model. On the target M5 Max, run:

```bash
graph-model mlx-doctor
graph-model mlx-doctor --load-model
```

A successful `--load-model` check is the release gate for the exact model repository, revision, installed MLX-LM architecture support, adapter, and Mac memory environment.
