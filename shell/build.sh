
env='BreakoutNoFrameskip-v4'
config='configs/build.yaml'
seed=0
total_timesteps=5000000
learning_starts=300
# --------------------------
# teacher
teacher_agent='dqn'
teacher_encoder='cnn'

# --------------------------
# distilllation
distil_encoder='one_cnn'
distil_agent='simple'
distil_method='kl'  # kl mse 

acceptance_ratio=1.0

python src/off_policy/run_train.py \
    --env-id $env \
    --seed $seed \
    --total-timesteps $total_timesteps \
    --learning-starts $learning_starts \
    --config $config \
    --teacher-agent $teacher_agent \
    --teacher-encoder $teacher_encoder \
    --distil-encoder $distil_encoder \
    --distil-agent $distil_agent \
    --distil-method $distil_method \
    --acceptance-ratio $acceptance_ratio 