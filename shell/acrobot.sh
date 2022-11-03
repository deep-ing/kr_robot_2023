
env='BreakoutNoFrameskip-v4'
env='Acrobot-v1'
# env='CartPole-v1'
config='configs/acrobot.yaml'
seed=2
total_timesteps=100000
learning_starts=5000
# --------------------------
# teacher
teacher_agent='dqn'
teacher_encoder='mlp_deep'
# --------------------------
# distilllation
distil_encoder='mlp_simple'
distil_agent='simple'
distil_method='mse' #'kl'  # kl mse 

acceptance_ratio=0.25

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