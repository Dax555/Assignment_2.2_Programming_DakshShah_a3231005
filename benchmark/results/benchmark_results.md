**Environment**
- python: 3.13.18
- numpy: 2.5.3
- platform: Linux-6.18.44-fc-v378-x86_64-with-glibc2.39
- processor: x86_64
- cpu_count: 2
- seed: 0

| case | n | baseline | loop (ms) | vectorised (ms) | speed-up | results match |
|---|---:|---|---:|---:|---:|:---:|
| StandardScaler | 1,000 | python loops | 1.070 | 0.148 | 7.2x | yes |
| StandardScaler | 10,000 | python loops | 12.738 | 0.864 | 14.8x | yes |
| StandardScaler | 100,000 | python loops | 157.673 | 10.647 | 14.8x | yes |
| MinMaxScaler | 1,000 | python loops | 0.814 | 0.233 | 3.5x | yes |
| MinMaxScaler | 10,000 | python loops | 8.829 | 1.171 | 7.5x | yes |
| MinMaxScaler | 100,000 | python loops | 129.350 | 17.161 | 7.5x | yes |
| OneHotEncoder | 1,000 | dict + loop | 0.120 | 0.432 | 0.3x | yes |
| OneHotEncoder | 10,000 | dict + loop | 1.198 | 2.754 | 0.4x | yes |
| OneHotEncoder | 100,000 | dict + loop | 17.812 | 27.435 | 0.6x | yes |
| rank (average, ties) | 1,000 | sort + tie loop | 0.272 | 0.188 | 1.4x | yes |
| rank (average, ties) | 10,000 | sort + tie loop | 2.736 | 0.885 | 3.1x | yes |
| rank (average, ties) | 100,000 | sort + tie loop | 55.279 | 9.813 | 5.6x | yes |
| percentile (5 levels) | 1,000 | sorted list | 0.091 | 0.262 | 0.3x | yes |
| percentile (5 levels) | 10,000 | sorted list | 1.011 | 0.363 | 2.8x | yes |
| percentile (5 levels) | 100,000 | sorted list | 12.953 | 0.943 | 13.7x | yes |
| topk (k=10) | 1,000 | heapq.nlargest | 0.034 | 0.060 | 0.6x | yes |
| topk (k=10) | 100,000 | heapq.nlargest | 1.385 | 0.461 | 3.0x | yes |
| topk (k=10) | 1,000,000 | heapq.nlargest | 13.864 | 7.150 | 1.9x | yes |
| confusion_matrix (k=5) | 1,000 | dict counting | 0.049 | 0.104 | 0.5x | yes |
| confusion_matrix (k=5) | 100,000 | dict counting | 4.535 | 1.335 | 3.4x | yes |
| confusion_matrix (k=5) | 1,000,000 | dict counting | 44.520 | 23.917 | 1.9x | yes |
| roc_auc_score (ties) | 200 | pairwise O(P*N) | 0.225 | 0.201 | 1.1x | yes |
| roc_auc_score (ties) | 1,000 | pairwise O(P*N) | 5.513 | 0.199 | 27.7x | yes |
| roc_auc_score (ties) | 3,000 | pairwise O(P*N) | 50.680 | 0.321 | 157.8x | yes |
| mean+var (5% NaN) | 1,000 | python loops | 0.824 | 0.308 | 2.7x | yes |
| mean+var (5% NaN) | 10,000 | python loops | 7.894 | 2.058 | 3.8x | yes |
| mean+var (5% NaN) | 100,000 | python loops | 81.281 | 23.669 | 3.4x | yes |
| histogram (20 bins) | 1,000 | bisect per element | 0.138 | 0.101 | 1.4x | yes |
| histogram (20 bins) | 100,000 | bisect per element | 12.466 | 3.110 | 4.0x | yes |
| histogram (20 bins) | 1,000,000 | bisect per element | 122.809 | 35.069 | 3.5x | yes |
| grad (central, n vars) | 100 | python-list f | 0.561 | 0.816 | 0.7x | yes |
| grad (central, n vars) | 500 | python-list f | 10.515 | 3.543 | 3.0x | yes |
| grad (central, n vars) | 2,000 | python-list f | 166.134 | 26.274 | 6.3x | yes |
| binary_search (100 queries) | 1,000 | linear scan | 1.611 | 0.425 | 3.8x | yes |
| binary_search (100 queries) | 10,000 | linear scan | 14.314 | 0.457 | 31.3x | yes |
| binary_search (100 queries) | 100,000 | linear scan | 151.901 | 0.520 | 292.2x | yes |
| binary_search method=loop vs searchsorted | 10,000 | library loop | 0.467 | 0.443 | 1.1x | yes |
| binary_search method=loop vs searchsorted | 1,000,000 | library loop | 0.703 | 0.532 | 1.3x | yes |
