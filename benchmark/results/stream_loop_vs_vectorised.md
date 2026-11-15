**Environment**
- python: 3.12.3
- numpy: 2.5.4
- platform: Linux-7.0.0-42-generic-x86_64-with-glibc2.39
- processor: x86_64
- cpu_count: 12
- seed: 0

| case | n | baseline | loop (ms) | vectorised (ms) | speed-up | results match |
|---|---:|---|---:|---:|---:|:---:|
| tree: best split of one node (d=8) | 200 | loop over features x thresholds x rows | 18.101 | 0.706 | 25.6x | yes |
| tree: best split of one node (d=8) | 1,000 | loop over features x thresholds x rows | 109.930 | 2.905 | 37.8x | yes |
| tree: best split of one node (d=8) | 4,000 | loop over features x thresholds x rows | 680.552 | 21.558 | 31.6x | yes |
| tree: predict (depth 6) | 2,000 | one root-to-leaf walk per row | 0.730 | 0.753 | 1.0x | yes |
| tree: predict (depth 6) | 20,000 | one root-to-leaf walk per row | 6.754 | 2.967 | 2.3x | yes |
| tree: predict (depth 6) | 200,000 | one root-to-leaf walk per row | 141.909 | 67.750 | 2.1x | yes |
| tree: buffered stream (buffer 500, depth 4) | 1,000 | loop tree, per-row predict | 288.336 | 19.860 | 14.5x | yes |
| tree: buffered stream (buffer 500, depth 4) | 4,000 | loop tree, per-row predict | 832.910 | 56.943 | 14.6x | yes |
| metrics: confusion matrix, chunks of 100 | 1,000,000 | one increment per sample | 97.001 | 727.256 | 0.1x | yes |
| stats: running mean/variance, chunks of 100 | 1,000,000 | Welford, one value at a time | 138.619 | 898.136 | 0.2x | yes |
| stats: histogram (20 bins), chunks of 100 | 1,000,000 | one bin lookup per value | 490.930 | 358.476 | 1.4x | yes |
| metrics: confusion matrix, chunks of 10,000 | 1,000,000 | one increment per sample | 87.276 | 34.157 | 2.6x | yes |
| stats: running mean/variance, chunks of 10,000 | 1,000,000 | Welford, one value at a time | 139.470 | 23.794 | 5.9x | yes |
| stats: histogram (20 bins), chunks of 10,000 | 1,000,000 | one bin lookup per value | 397.458 | 50.056 | 7.9x | yes |
