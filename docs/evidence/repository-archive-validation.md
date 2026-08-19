# Public repository archive validation

Validated before packaging:

- 12 Git commits
- 11 annotated release tags: `v0.1.0` through `v0.5.6`
- Every tag matches every regular file in the corresponding original source ZIP byte-for-byte
- The only unrepresentable source artifact is an empty build directory in v0.5.4; Git does not track empty directories
- 30 preserved release assets verified against normalized SHA-256 manifests
- Active public source/docs tree scanned for local user paths, withdrawn application language, and common credential-token shapes
- Python source, tests, and scripts compiled successfully
- Default graph validation passed:
  - graph: `coding-supergraph`
  - graph version: `0.3.0`
  - nodes: 12
  - edges: 19
  - terminals: `abort`, `finish`
  - schema: `1536e86a64cbf09a1c0308ab72985a67eba47eb748f3b4f7ba23accc25e7543f`
- Git object database passed `git fsck --full --strict`

The original release artifacts are intentionally byte-for-byte copies. Current documentation and sanitized terminal breadcrumbs are layered onto `main` after the `v0.5.6` source tag.
