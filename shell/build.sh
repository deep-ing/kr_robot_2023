
env='BreakoutNoFrameskip-v4'
config='configs/build.yaml'
seed=0
total_timesteps=10000000
teacher_encoder='cnn'
student_encoder='cnn'
learning_starts=20000

python src/c51/run_train.py \
    --env-id $env \
    --seed $seed \
    --total-timesteps $total_timesteps \
    --learning-starts $learning_starts \
    --config $config \
    --teacher-encoder $teacher_encoder \
    --student-encoder $student_encoder