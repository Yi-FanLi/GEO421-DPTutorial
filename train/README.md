# Train a machine learning potential for water

This part of the tutorial is to train a machine learning potential for water. For explanations of the `input.json` file and the dataset, please refer to the slides.

`run.slurm` is a slurm script to submit the training task on della. The line
```
( /bin/sh -c '{ if [ ! -f model.ckpt.index ]; then dp train input.json; else dp train input.json --restart model.ckpt; fi }'&&dp freeze&&dp compress ) 1>>train.log 2>>train.log
```
is to check if the training is a new one or a restart. If it is a restart, the training will continue from the last checkpoint. If it is a new one, the training will start from scratch.

Please run 
```
sbatch run.slurm
```
on della to submit the training task. You can check the status of the task by
```
squeue
```

After training, you may use the `infer.py` script to perform inference with the trained model. 
```
conda activate /tigress/yifanl/usr/licensed/anaconda3/2021.11/envs/dp-v2.2.7
python infer.py
```
```
After that, you may use the `train.ipynb` notebook to check the error of the model.
