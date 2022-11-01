
env='BreakoutNoFrameskip-v4'
config='configs/build.yaml'
seed=0
total_timesteps=5000000
teacher_agent='dqn'
distil_agent='simple'
teacher_encoder='cnn'
distil_encoder='one_cnn'
learning_starts=2000

python src/off_policy/run_train.py \
    --env-id $env \
    --seed $seed \
    --total-timesteps $total_timesteps \
    --learning-starts $learning_starts \
    --config $config \
    --teacher-agent $teacher_agent \
    --teacher-encoder $teacher_encoder \
    --distil-encoder $distil_encoder \
    --distil-agent $distil_agent 