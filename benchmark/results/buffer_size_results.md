**Environment**
- python: 3.12.3
- numpy: 2.5.4
- platform: Linux-7.0.0-42-generic-x86_64-with-glibc2.39
- processor: x86_64
- cpu_count: 12
- seed: 0
- stream: 40,000 samples, 10 features, 4 classes, chunks of 100, labelling rule changes at the half-way point, 5% label noise
- tree: max_depth=6, gini; accuracy is prequential (test-then-train)
- samples to recover: samples after the drift until the rolling accuracy (last 1,000 samples) is back at 0.8 or more, checked from 1,000 samples after the drift (when the window holds only post-drift data)

| buffer_size | retrains | accuracy | before drift | after drift | samples to recover | mean retrain (ms) | total retrain (ms) | peak memory (KiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 400 | 0.900 | 0.905 | 0.896 | 1,000 | 3.60 | 1440.2 | 2.9 |
| 250 | 133 | 0.911 | 0.917 | 0.904 | 1,200 | 12.49 | 1660.7 | 23.5 |
| 500 | 80 | 0.917 | 0.926 | 0.908 | 1,300 | 18.36 | 1468.8 | 40.3 |
| 1,000 | 40 | 0.924 | 0.941 | 0.909 | 1,700 | 26.88 | 1075.2 | 83.6 |
| 2,000 | 20 | 0.913 | 0.950 | 0.880 | 2,700 | 44.85 | 896.9 | 170.3 |
| 5,000 | 8 | 0.856 | 0.957 | 0.781 | 5,700 | 104.85 | 838.8 | 427.2 |
| 10,000 | 4 | 0.722 | 0.960 | 0.604 | 10,700 | 203.36 | 813.4 | 856.6 |
