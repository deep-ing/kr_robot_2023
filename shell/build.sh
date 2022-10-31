
env='leaper'
config='configs/build.yaml'
seed=0
total_timesteps=1000000
teacher_encoder='resnet'
student_encoder='resnet'

python src/c51/run_train.py \
    --env-id $env \
    --seed $seed \
    --total-timesteps $total_timesteps \
    --learning-starts 1000 \
    --config $config \
    --teacher-encoder $teacher_encoder \
    --student-encoder $student_encoder