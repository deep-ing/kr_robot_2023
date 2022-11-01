
env='BreakoutNoFrameskip-v4'
config='configs/build.yaml'
seed=0
total_timesteps=1000000
teacher_agent='dqn'
teacher_encoder='cnn'
student_encoder='cnn'
learning_starts=400

python src/off_policy/run_train.py \
    --env-id $env \
    --seed $seed \
    --total-timesteps $total_timesteps \
    --learning-starts $learning_starts \
    --config $config \
    --teacher-agent $teacher_agent \
    --teacher-encoder $teacher_encoder \
    --student-encoder $student_encoder