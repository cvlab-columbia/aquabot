#!/bin/bash

# Define the commands to run
commands=(
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 1 --n_pred 8 --interval 100 --batch_size 32 --learning_rate 0.001 --loss_function mse"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 2 --n_pred 8 --interval 100 --batch_size 32 --learning_rate 0.001 --loss_function mse"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 3 --n_pred 8 --interval 100 --batch_size 32 --learning_rate 0.001 --loss_function mse"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 1 --n_pred 8 --interval 100 --batch_size 64 --learning_rate 0.001 --loss_function mse"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 2 --n_pred 8 --interval 100 --batch_size 64 --learning_rate 0.001 --loss_function mse"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 3 --n_pred 8 --interval 100 --batch_size 64 --learning_rate 0.001 --loss_function mse"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 1 --n_pred 8 --interval 100 --batch_size 32 --learning_rate 0.0001 --loss_function mse"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 2 --n_pred 8 --interval 100 --batch_size 32 --learning_rate 0.0001 --loss_function mse"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 3 --n_pred 8 --interval 100 --batch_size 32 --learning_rate 0.0001 --loss_function mse"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 1 --n_pred 8 --interval 100 --batch_size 32 --learning_rate 0.001 --loss_function l1"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 2 --n_pred 8 --interval 100 --batch_size 32 --learning_rate 0.001 --loss_function l1"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 3 --n_pred 8 --interval 100 --batch_size 32 --learning_rate 0.001 --loss_function l1"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 1 --n_pred 8 --interval 100 --batch_size 64 --learning_rate 0.001 --loss_function l1"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 2 --n_pred 8 --interval 100 --batch_size 64 --learning_rate 0.001 --loss_function l1"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 3 --n_pred 8 --interval 100 --batch_size 64 --learning_rate 0.001 --loss_function l1"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 1 --n_pred 8 --interval 100 --batch_size 32 --learning_rate 0.0001 --loss_function l1"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 2 --n_pred 8 --interval 100 --batch_size 32 --learning_rate 0.0001 --loss_function l1"
    "python train.py --root_dir ../fifish-vevo/rov_control/data_0806 --num_epochs 40 --n_obs 3 --n_pred 8 --interval 100 --batch_size 32 --learning_rate 0.0001 --loss_function l1"
)
#!/bin/bash

# Run every two commands in parallel
for ((i=0; i<${#commands[@]}; i+=2)); do
    if [ $((i + 1)) -lt ${#commands[@]} ]; then
        echo "Running: ${commands[$i]}"
        echo "Running: ${commands[$((i + 1))]}"
        eval "${commands[$i]}" & eval "${commands[$((i + 1))]}" &
    else
        echo "Running: ${commands[$i]}"
        eval "${commands[$i]}" &
    fi
    wait
done

echo "All commands have been executed."