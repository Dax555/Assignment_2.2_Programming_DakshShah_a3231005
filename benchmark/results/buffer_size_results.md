**Environment**
- python: 3.13.18
- numpy: 2.5.3
- platform: Linux-6.18.44-fc-v378-x86_64-with-glibc2.39
- processor: x86_64
- cpu_count: 2
- seed: 0
- stream: 40,000 samples, 10 features, 4 classes, chunks of 100, labelling rule changes at the half-way point, 5% label noise
- tree: max_depth=6, gini; accuracy is prequential (test-then-train)
- samples to recover: samples after the drift until the rolling accuracy (last 1,000 samples) is back at 0.8 or more, checked from 1,000 samples after the drift (when the window holds only post-drift data)

| buffer_size | retrains | accuracy | before drift | after drift | samples to recover | mean retrain (ms) | total retrain (ms) | peak memory (KiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 400 | 0.900 | 0.905 | 0.896 | 1,000 | 1.33 | 533.0 | 2.9 |
| 250 | 133 | 0.911 | 0.917 | 0.904 | 1,200 | 3.75 | 498.2 | 23.5 |
| 500 | 80 | 0.917 | 0.926 | 0.908 | 1,300 | 6.20 | 496.2 | 40.3 |
| 1,000 | 40 | 0.924 | 0.941 | 0.909 | 1,700 | 12.88 | 515.2 | 83.6 |
| 2,000 | 20 | 0.913 | 0.950 | 0.880 | 2,700 | 20.36 | 407.3 | 170.3 |
| 5,000 | 8 | 0.856 | 0.957 | 0.781 | 5,700 | 52.07 | 416.5 | 427.2 |
| 10,000 | 4 | 0.722 | 0.960 | 0.604 | 10,700 | 106.93 | 427.7 | 856.6 |
