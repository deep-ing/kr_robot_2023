env='BreakoutNoFrameskip-v4'
env='LunarLander-v2'
env='CartPole-v1'
config='configs/build.yaml'
seed=0
total_timesteps=200000
learning_starts=10000
# --------------------------
# teacher
teacher_agent='dqn'
teacher_encoder='mlp_deep'
# --------------------------
# distilllation
distil_encoder='mlp_simple'
distil_agent='simple'
distil_method='mse' #'kl'  # kl mse 

acceptance_ratio=1.0

tau=1.0
target_update_frequency=2000
dict='{"tau" : '$tau', "'$teacher_agent'/target_update_frequency":'$target_update_frequency'}'

start_config='configs/build.yaml'
save_config='configs/untracked/'$tau'_tuf_'$target_update_frequency'.yaml'
python src/modify_yaml.py \
    --items "$dict" \
    --config $start_config \
    --save-path $save_config




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