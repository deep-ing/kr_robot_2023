#!/bin/bash

env='BreakoutNoFrameskip-v4'
seed=0
total_timesteps=1000000
teacher_encoder='cnn'
student_encoder='cnn'
learning_starts=10000


target_update_frequency=10000
tau=0.1 # 0.1 0.05

dict='{"tau" : '$tau', "target_update_frequency":'$target_update_frequency'}'

start_config='configs/build.yaml'
save_config='configs/untracked/'$tau'_tuf_'$target_update_frequency'.yaml'
python src/modify_yaml.py \
    --items "$dict" \
    --config $start_config \
    --save-path $save_config

python src/c51/run_train.py \
    --env-id $env \
    --seed $seed \
    --total-timesteps $total_timesteps \
    --learning-starts $learning_starts \
    --config $save_config \
    --teacher-encoder $teacher_encoder \
    --student-encoder $student_encoder \
    --postfix 'hyp_tau_'$tau'_tuf_'$target_update_frequency \



