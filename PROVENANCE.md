# Provenance and public-data policy

This repository was reconstructed from the original source ZIPs, wheels, checksum manifests, release metadata, and terminal evidence produced during the project.

## Included

- Source history from v0.1.0 through v0.5.6
- An annotated Git tag for every release
- Original source archives and wheels
- Original checksum and release-metadata files where available
- A normalized checksum manifest for every archived release
- Sanitized terminal evidence preserving failures, fixes, and results
- Public summaries of the M5 qualification and bootstrap corpus

## Deliberately excluded

- Runtime SQLite databases
- Detached worktrees
- Raw prompts
- Raw hidden tensors
- Local model files and caches
- Credentials, tokens, private keys, and private repository data
- Machine-specific unsanitized paths and personal email addresses from the active source/docs tree

The release artifacts remain byte-for-byte copies and may therefore preserve historical example paths embedded in those original archives. Sanitization applies to the active public source/docs tree and to the terminal-transcript copies under `docs/journey/raw/`.
