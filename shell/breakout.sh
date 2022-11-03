
env='BreakoutNoFrameskip-v4'
# env='LunarLander-v2'
# env='CartPole-v1'
config='configs/breakout.yaml'
total_timesteps=1000000
learning_starts=20000
# --------------------------
# teacher
teacher_agent='dqn'
teacher_encoder='cnn'
# --------------------------
# distilllation
distil_encoder='one_cnn'
distil_agent='simple'
distil_method='mse' #'kl'  # kl mse 

acceptance_ratio=0.25

for seed in 0 1 2
do
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
done