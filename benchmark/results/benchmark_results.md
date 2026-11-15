**Environment**
- python: 3.12.3
- numpy: 2.5.4
- platform: Linux-7.0.0-42-generic-x86_64-with-glibc2.39
- processor: x86_64
- cpu_count: 12
- seed: 0

| case | n | baseline | loop (ms) | vectorised (ms) | speed-up | results match |
|---|---:|---|---:|---:|---:|:---:|
| StandardScaler | 1,000 | python loops | 2.763 | 0.322 | 8.6x | yes |
| StandardScaler | 10,000 | python loops | 34.487 | 2.502 | 13.8x | yes |
| StandardScaler | 100,000 | python loops | 403.410 | 21.885 | 18.4x | yes |
| MinMaxScaler | 1,000 | python loops | 1.763 | 0.661 | 2.7x | yes |
| MinMaxScaler | 10,000 | python loops | 23.414 | 2.077 | 11.3x | yes |
| MinMaxScaler | 100,000 | python loops | 407.646 | 25.774 | 15.8x | yes |
| OneHotEncoder | 1,000 | dict + loop | 1.144 | 2.023 | 0.6x | yes |
| OneHotEncoder | 10,000 | dict + loop | 3.743 | 5.883 | 0.6x | yes |
| OneHotEncoder | 100,000 | dict + loop | 60.083 | 58.746 | 1.0x | yes |
| rank (average, ties) | 1,000 | sort + tie loop | 0.460 | 0.291 | 1.6x | yes |
| rank (average, ties) | 10,000 | sort + tie loop | 5.431 | 1.365 | 4.0x | yes |
| rank (average, ties) | 100,000 | sort + tie loop | 78.274 | 14.850 | 5.3x | yes |
| percentile (5 levels) | 1,000 | sorted list | 0.140 | 0.602 | 0.2x | yes |
| percentile (5 levels) | 10,000 | sorted list | 1.704 | 0.679 | 2.5x | yes |
| percentile (5 levels) | 100,000 | sorted list | 22.878 | 3.084 | 7.4x | yes |
| topk (k=10) | 1,000 | heapq.nlargest | 0.097 | 0.147 | 0.7x | yes |
| topk (k=10) | 100,000 | heapq.nlargest | 2.243 | 0.917 | 2.4x | yes |
| topk (k=10) | 1,000,000 | heapq.nlargest | 36.375 | 14.626 | 2.5x | yes |
| confusion_matrix (k=5) | 1,000 | dict counting | 0.120 | 0.173 | 0.7x | yes |
| confusion_matrix (k=5) | 100,000 | dict counting | 12.427 | 2.987 | 4.2x | yes |
| confusion_matrix (k=5) | 1,000,000 | dict counting | 126.987 | 40.181 | 3.2x | yes |
| roc_auc_score (ties) | 200 | pairwise O(P*N) | 0.443 | 0.274 | 1.6x | yes |
| roc_auc_score (ties) | 1,000 | pairwise O(P*N) | 10.207 | 0.352 | 29.0x | yes |
| roc_auc_score (ties) | 3,000 | pairwise O(P*N) | 91.174 | 0.551 | 165.5x | yes |
| mean+var (5% NaN) | 1,000 | python loops | 1.797 | 0.612 | 2.9x | yes |
| mean+var (5% NaN) | 10,000 | python loops | 19.934 | 6.772 | 2.9x | yes |
| mean+var (5% NaN) | 100,000 | python loops | 195.659 | 40.131 | 4.9x | yes |
| histogram (20 bins) | 1,000 | bisect per element | 0.432 | 0.133 | 3.3x | yes |
| histogram (20 bins) | 100,000 | bisect per element | 44.036 | 4.570 | 9.6x | yes |
| histogram (20 bins) | 1,000,000 | bisect per element | 580.114 | 57.986 | 10.0x | yes |
| grad (central, n vars) | 100 | python-list f | 2.361 | 1.626 | 1.5x | yes |
| grad (central, n vars) | 500 | python-list f | 27.042 | 9.312 | 2.9x | yes |
| grad (central, n vars) | 2,000 | python-list f | 430.534 | 50.854 | 8.5x | yes |
| binary_search (100 queries) | 1,000 | linear scan | 2.959 | 0.997 | 3.0x | yes |
| binary_search (100 queries) | 10,000 | linear scan | 31.395 | 1.050 | 29.9x | yes |
| binary_search (100 queries) | 100,000 | linear scan | 340.171 | 1.065 | 319.4x | yes |
| binary_search method=loop vs searchsorted | 10,000 | library loop | 1.054 | 1.039 | 1.0x | yes |
| binary_search method=loop vs searchsorted | 1,000,000 | library loop | 1.273 | 1.097 | 1.2x | yes |
