# ECPI(The contents of the repository are currently under development, with an official release planned for the future.)

## Benchmark Results (4 seeds)

| task                       | BC   | CQL   | TD3+BC | EDAC  | IQL   | DARL  | MOPO  | COMBO | ROMI  | RAMBO | MOBILE | ECPI   |
|----------------------------|------|-------|--------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| halfcheetah-random         | 2.2  | 31.3  | 11     | 28.4  | 11.2  | 32.4  | 34    | 38.8  | 24.5  | 40    | 39.3   | 41.4   |
| hopper-random              | 3.7  | 5.3   | 8.5    | 25.3  | 7.9   | 32.3  | 31.7  | 17.9  | 30.2  | 21.6  | 31.9   | 33.3   |
| walker2d-random            | 1.3  | 5.4   | 1.6    | 16.6  | 5.9   | 21.7  | 7.4   | 7     | 7.5   | 11.5  | 17.9   | 17.9   |
| halfcheetah-medium         | 43.2 | 46.9  | 48.3   | 65.9  | 47.4  | 69.8  | 73.3  | 54.2  | 49.1  | 77.6  | 74.6   | 82.2   |
| hopper-medium              | 54.1 | 61.9  | 59.3   | 101.6 | 66.2  | 63.7  | 62.8  | 97.2  | 72.3  | 92.8  | 106.6  | 107.59 |
| walker2d-medium            | 70.9 | 79.5  | 83.7   | 92.5  | 78.3  | 84.5  | 84.1  | 81.9  | 84.3  | 86.9  | 87.7   | 91.6   |
| halfcheetah-medium-replay  | 37.6 | 45.3  | 44.6   | 61.3  | 44.2  | 59.6  | 72.1  | 55.1  | 47    | 68.9  | 71.7   | 73.6   |
| hopper-medium-replay       | 16.6 | 86.3  | 60.9   | 101   | 94.7  | 96.7  | 103.5 | 89.5  | 98.1  | 96.6  | 103.9  | 110.6  |
| walker2d-medium-replay     | 20.3 | 76.8  | 81.8   | 87.1  | 73.8  | 99.4  | 86.6  | 56.1  | 108.7 | 85    | 89.9   | 94.2   |
| halfcheetah-medium-expert  | 44   | 95    | 90.7   | 106.3 | 86.7  | 95.7  | 90.8  | 90.2  | 86.8  | 93.7  | 108.2  | 110.7  |
| hopper-medium-expert       | 53.9 | 96.9  | 98     | 110.7 | 91.5  | 110.6 | 81.6  | 111.1 | 111.4 | 83.3  | 112.6  | 115.2  |
| walker2d-medium-expert     | 90.1 | 109.1 | 110.1  | 114.7 | 109.6 | 110   | 112.9 | 103.3 | 109.7 | 68.3  | 115.2  | 116.5  |


### Installation

- python 3.8.5 and cuda 9.2
- Mujoco210
- Mujoco_py
- D4RL, NeoRL
- Installed using `/requirements.txt`

## Training

 `/run_example/run_bupo2.py `