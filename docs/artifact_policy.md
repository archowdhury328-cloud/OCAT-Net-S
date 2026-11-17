# Artifact policy
| Artifact | Location | Git |
|---|---|---|
| Source, configs, tests, docs, small aggregate tables | repo | yes |
| Raw/derived images, caches | `data/`, `cache/` | never |
| Manifests (file hashes, names), per-image predictions | `manifests/private/`, `runs/` | never |
| Checkpoints / bundles | release assets or external storage with SHA-256, only if data licences permit | no |
| Secrets | environment / credential manager | never |
Code licence (MIT) does not cover datasets, third-party weights, or the paper (CC BY 4.0). Public availability of a dataset is not redistribution consent.
