
import numpy as np
import Evaluation.resPlot as resPlot

path_log = input("Enter the root path where path_log file exists >>>")
f_ptr = open(f"{path_log}/path_log", mode = "r")
paths = [
    i[:-1] for i in f_ptr
]
paths = paths[1:]

for i in paths:
    print(i)

test_acc = []
test_adv_acc = []
for i in paths:
    test_acc.append(np.load(f"{i}/eval/all_acc.npy"))
    test_adv_acc.append(np.load(f"{i}/eval/all_adv_acc.npy"))

baseline_path1 = input("Enter the path of baseline model >>>")
baseline_path2 = input("Enter the path of baseline model >>>")

test_acc.append(np.load(f"{baseline_path1}/eval/all_acc.npy"))
test_adv_acc.append(baseline_adv_acc = np.load(f"{baseline_path1}/eval/all_adv_acc.npy"))
test_acc.append(np.load(f"{baseline_path2}/eval/all_acc.npy"))
test_adv_acc.append(baseline_adv_acc = np.load(f"{baseline_path2}/eval/all_adv_acc.npy"))

xaxis = [1, 5, 10, 15, 20, 30]
