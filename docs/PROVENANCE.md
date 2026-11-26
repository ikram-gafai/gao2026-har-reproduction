# Commit provenance

The repository history records the order in which the work was frozen:

| Commit | Stage | Content |
|:--|:--|:--|
| [`ab93dfe1`](https://github.com/ikram-gafai/gao2026-har-reproduction/commit/ab93dfe16baa189523916476288cfe56fe4cc406) | Pre-registration | Protocol, frozen configuration, data pipeline and feature mapping, committed before any model was fitted |
| [`2678acad`](https://github.com/ikram-gafai/gao2026-har-reproduction/commit/2678acadd135a4bc751a4c32e1d279976b2a3c51) | Implementation | Training, evaluation, bootstrap and repeatability code that produced the results |
| [`be86b6d2`](https://github.com/ikram-gafai/gao2026-har-reproduction/commit/be86b6d2ad4c1673cbbd6864377e413d8272e373) | Results | Unedited outputs of the 20 pre-registered runs, plus the `verify-results` audit command |

## Recreated commit identifiers

Before the repository was first published, these three commits were recreated to correct their author metadata.
- Their file trees are byte-identical to the originals.
- Their author and commit timestamps are preserved.
- Only the commit identifiers changed.

`results/manifest.json` was written by the experiment run and is never edited, so it still cites the original implementation identifier.

| Commit | Original identifier | File tree (unchanged) |
|:--|:--|:--|
| `ab93dfe1` protocol | `e14633759ae9a82b51617e70fd2613ff2059901e` | `ad4a3e82206f0593669a7556c950a91d6c9cfc58` |
| `2678acad` implementation | `ca2b88017070d0bdbbbfb02129b7ffc60427edf3` (cited in `results/manifest.json`) | `c798fc82ee8eb64d45a9c63a16d6fea51e0cca01` |
| `be86b6d2` saved results | `6039ca6b1bfe78c8dd47cff21943c8037cd10517` | `9fd9c4cf357759a40ccda3809a90f529012e8ea5` |

## How to verify

A commit's tree hash identifies its exact file contents. Equal tree hashes therefore mean identical code, configuration and results.

```bash
git rev-parse "ab93dfe1^{tree}"   # ad4a3e82206f0593669a7556c950a91d6c9cfc58
git rev-parse "2678acad^{tree}"   # c798fc82ee8eb64d45a9c63a16d6fea51e0cca01
git rev-parse "be86b6d2^{tree}"   # 9fd9c4cf357759a40ccda3809a90f529012e8ea5
```

The tree checks confirm what each commit contains. Separately, `python -m har_repro.cli verify-results` recomputes every metric, the participant bootstrap and the outcome decision from the committed predictions.
