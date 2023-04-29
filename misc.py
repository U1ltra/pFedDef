import os

scale_set = [scale for scale in range(10, 200, 10)] + [scale for scale in range(210, 240, 1)]
scale_set.sort()
exp_names = [f'rep_scale{int(i)}' for i in scale_set]
exp_root_path = f"/home/ubuntu/Documents/jiarui/experiments/FedAvg_adv/rep_atk"
path_log = open(f"{exp_root_path}/path_log", mode = "w")

path_log.write(f'FedAvg\n')

for i, exp_name in enumerate(exp_names):
    path_log.write(f'{exp_root_path}/{exp_name}\n')
    print(f'{exp_root_path}/{exp_name}')
    if not os.path.exists(f'{exp_root_path}/{exp_name}'):
        print(f'{exp_root_path}/{exp_name}')

path_log.close()

