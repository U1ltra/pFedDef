import numpy as np

atk_nums = [1, 3, 5, 10, 15, 20, 25]
client_weights = np.load(
    "/home/ubuntu/Documents/jiarui/experiments/multi_atker/client_weights.npy"
)
best_scales = [int(1 / client_weights[0:atk_num].sum()) for atk_num in atk_nums]
params = []
for idx, scale in enumerate(best_scales):
    print(idx, scale)
    if scale - 10 <= 0:
        for i in range(1, scale+1):
            params.append((atk_nums[idx], i))
            print(atk_nums[idx], i)
    else:
        for i in range(0, 10, 2):
            params.append((atk_nums[idx], scale + i - 8))
            print(atk_nums[idx], scale + i - 8)
    print()
