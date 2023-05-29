### FedUnlearn
#### Rerun the FAT training from data generation. Record the steps. 

``` bash
cd /path/to/repo
cd data/cifar10
python generate_data.py --n_tasks 40 --n_components 3 --alpha 0.4 --s_frac 1.0 --tr_frac 0.8 --seed 12345
```

```bash
CODE_PATH=/path/to/repo
EXP_PATH=/path/to/log

EXP_NAME=FAT_baseline

echo -e "\nstart training"
echo "${EXP_PATH}" | python ${CODE_PATH}/run_experiment_pFedDef.py > ${EXP_PATH}/training_log

echo -e "\nstart evaluation..."
echo "${EXP_PATH}" | python ${CODE_PATH}/eval_acc_io_path.py > ${EXP_PATH}/eval_log
```

|      Epochs        | acc               |    adv acc        | 
| ------------------ | ----------------- | ----------------- | 
| 150                | 79.604%           | 60.240%           |
| 200                | 78.833%           | 64.837%           |
| 250                | 78.856%           | 66.192%           |
