# Rail branch audit, 19 September 2026

Remote references were refreshed and reviewed without changing the checkout or merging branches.

| Branch | Commit | Finding |
|---|---|---|
| `origin/xs-experiments` and local `xs-experiments` | `e7303a00b6576658e1471053ef84b64cb90fdbe8` | Historical incumbent and model/window experiments. Thresholds were selected on outer predictions, so reported scores are development results rather than nested estimates. |
| `origin/sg-experiments` | `fa07703e84d8148acff3543bcafa654d3bf63e4f` | Its full spectral extractor is already included in the current SG representation. An additional narrow paired-position feature combination remains a possible follow-up. |
| `origin/rail-corrugation/exp-20260918-reproducible-campaign` | `7aeaef314beb4496ab7713a74c33505c04e3e94a` | Best recorded candidate: 0.8087 stratified / 0.7817 blocked. Speed-removal, weighting and reference changes already received local duplicate-safe retesting. |
| Local `codex/rail-branch-review` | `10289606191d10a03138b3bdb14c48e9466aeda2` | Contains the later reference/feature audit and negative campaign results, already covered by the current research. |

## Corrections to historical interpretation

XS `Rail Corrugation/notebooks/autoresearch2.py`, `run_mirror()`, adds swapped feature rows `[r2, r1]` but keeps the original label order. The resulting contradictory labels mean its reported loss does not establish that the physical sides are non-exchangeable. Current canonical paired-model tests verify both feature and label transformations.

The dedicated Rail campaign groups duplicate recordings for blocked outer folds, but its ordinary inner stratified folds and other stratified splits can still separate duplicates. Current outer and inner duplicate grouping remains the comparison protocol.

Screening and confirmation partitions reuse the same 272 recordings, including only 14 Side I faults. None of these branch comparisons is a fresh independent test, and branch scores from different protocols should not be compared as if matched.

## Remaining hypothesis

If needed, test only the paired-position spectral contrasts already calculated by SG's `aggregate(..., paired=True)` alongside legacy+SG RF features, excluding side identity. Current legacy selection omits `own_pair_` and `rel_pair_` columns. The earlier broad paired-feature ET result (0.8010 versus 0.8232 baseline) does not test this narrower RF+SG combination, but provides no positive evidence that it will help. It is a possible experiment, not a stronger model available for immediate submission.

No stronger validated branch model was identified for direct import. The forest-search candidate retains the XS-derived reference features and SG-derived representation, and is assessed with the current matched validation rather than historical branch scores.
