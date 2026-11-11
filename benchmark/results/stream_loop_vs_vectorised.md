**Environment**
- python: 3.13.18
- numpy: 2.5.3
- platform: Linux-6.18.44-fc-v378-x86_64-with-glibc2.39
- processor: x86_64
- cpu_count: 2
- seed: 0

| case | n | baseline | loop (ms) | vectorised (ms) | speed-up | results match |
|---|---:|---|---:|---:|---:|:---:|
| tree: best split of one node (d=8) | 200 | loop over features x thresholds x rows | 10.055 | 0.396 | 25.4x | yes |
| tree: best split of one node (d=8) | 1,000 | loop over features x thresholds x rows | 55.039 | 1.325 | 41.5x | yes |
| tree: best split of one node (d=8) | 4,000 | loop over features x thresholds x rows | 266.248 | 4.592 | 58.0x | yes |
| tree: predict (depth 6) | 2,000 | one root-to-leaf walk per row | 0.394 | 0.188 | 2.1x | yes |
| tree: predict (depth 6) | 20,000 | one root-to-leaf walk per row | 3.780 | 1.123 | 3.4x | yes |
| tree: predict (depth 6) | 200,000 | one root-to-leaf walk per row | 57.380 | 22.145 | 2.6x | yes |
| tree: buffered stream (buffer 500, depth 4) | 1,000 | loop tree, per-row predict | 113.258 | 6.178 | 18.3x | yes |
| tree: buffered stream (buffer 500, depth 4) | 4,000 | loop tree, per-row predict | 438.776 | 22.032 | 19.9x | yes |
| metrics: confusion matrix, chunks of 100 | 1,000,000 | one increment per sample | 39.042 | 190.566 | 0.2x | yes |
| stats: running mean/variance, chunks of 100 | 1,000,000 | Welford, one value at a time | 48.047 | 303.379 | 0.2x | yes |
| stats: histogram (20 bins), chunks of 100 | 1,000,000 | one bin lookup per value | 88.795 | 136.912 | 0.6x | yes |
| metrics: confusion matrix, chunks of 10,000 | 1,000,000 | one increment per sample | 38.377 | 16.305 | 2.4x | yes |
| stats: running mean/variance, chunks of 10,000 | 1,000,000 | Welford, one value at a time | 50.000 | 8.518 | 5.9x | yes |
| stats: histogram (20 bins), chunks of 10,000 | 1,000,000 | one bin lookup per value | 79.905 | 33.913 | 2.4x | yes |
