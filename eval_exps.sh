#!/bin/sh

# freq_dependency
python main.py --ftype 0 --model_dir exp/freq_dependency --data_type sine_0.5Hz_10deg_0offset --eval_type sine_1Hz_10deg_0offset &
python main.py --ftype 1 --model_dir exp/freq_dependency --data_type sine_0.5Hz_10deg_0offset --eval_type sine_1Hz_10deg_0offset &
python main.py --ftype 2 --model_dir exp/freq_dependency --data_type sine_0.5Hz_10deg_0offset --eval_type sine_1Hz_10deg_0offset &
wait
python main.py --ftype 0 --model_dir exp/freq_dependency --data_type sine_0.5Hz_10deg_0offset --eval_type sine_5Hz_10deg_0offset &
python main.py --ftype 1 --model_dir exp/freq_dependency --data_type sine_0.5Hz_10deg_0offset --eval_type sine_5Hz_10deg_0offset &
python main.py --ftype 2 --model_dir exp/freq_dependency --data_type sine_0.5Hz_10deg_0offset --eval_type sine_5Hz_10deg_0offset &
wait

python main.py --ftype 0 --model_dir exp/freq_dependency --data_type sine_5Hz_10deg_0offset --eval_type sine_1Hz_10deg_0offset &
python main.py --ftype 1 --model_dir exp/freq_dependency --data_type sine_5Hz_10deg_0offset --eval_type sine_1Hz_10deg_0offset &
python main.py --ftype 2 --model_dir exp/freq_dependency --data_type sine_5Hz_10deg_0offset --eval_type sine_1Hz_10deg_0offset &
wait
python main.py --ftype 0 --model_dir exp/freq_dependency --data_type sine_5Hz_10deg_0offset --eval_type sine_5Hz_10deg_0offset &
python main.py --ftype 1 --model_dir exp/freq_dependency --data_type sine_5Hz_10deg_0offset --eval_type sine_5Hz_10deg_0offset &
python main.py --ftype 2 --model_dir exp/freq_dependency --data_type sine_5Hz_10deg_0offset --eval_type sine_5Hz_10deg_0offset &
wait

# amp_denpendency
python main.py --ftype 0 --model_dir exp/amp_dependency --data_type sine_1Hz_10deg_0offset --eval_type sine_1Hz_20deg_0offset &
python main.py --ftype 1 --model_dir exp/amp_dependency --data_type sine_1Hz_10deg_0offset --eval_type sine_1Hz_20deg_0offset &
python main.py --ftype 2 --model_dir exp/amp_dependency --data_type sine_1Hz_10deg_0offset --eval_type sine_1Hz_20deg_0offset &
wait
python main.py --ftype 0 --model_dir exp/amp_dependency --data_type sine_1Hz_10deg_0offset --eval_type sine_1Hz_10deg_-40offset &
python main.py --ftype 1 --model_dir exp/amp_dependency --data_type sine_1Hz_10deg_0offset --eval_type sine_1Hz_10deg_-40offset &
python main.py --ftype 2 --model_dir exp/amp_dependency --data_type sine_1Hz_10deg_0offset --eval_type sine_1Hz_10deg_-40offset &
wait
python main.py --ftype 0 --model_dir exp/amp_dependency --data_type sine_1Hz_10deg_0offset --eval_type sine_1Hz_10deg_40offset &
python main.py --ftype 1 --model_dir exp/amp_dependency --data_type sine_1Hz_10deg_0offset --eval_type sine_1Hz_10deg_40offset &
python main.py --ftype 2 --model_dir exp/amp_dependency --data_type sine_1Hz_10deg_0offset --eval_type sine_1Hz_10deg_40offset &
wait

python main.py --ftype 0 --model_dir exp/amp_dependency --data_type sine_1Hz_50deg_0offset --eval_type sine_1Hz_20deg_0offset &
python main.py --ftype 1 --model_dir exp/amp_dependency --data_type sine_1Hz_50deg_0offset --eval_type sine_1Hz_20deg_0offset &
python main.py --ftype 2 --model_dir exp/amp_dependency --data_type sine_1Hz_50deg_0offset --eval_type sine_1Hz_20deg_0offset &
wait
python main.py --ftype 0 --model_dir exp/amp_dependency --data_type sine_1Hz_50deg_0offset --eval_type sine_1Hz_10deg_-40offset &
python main.py --ftype 1 --model_dir exp/amp_dependency --data_type sine_1Hz_50deg_0offset --eval_type sine_1Hz_10deg_-40offset &
python main.py --ftype 2 --model_dir exp/amp_dependency --data_type sine_1Hz_50deg_0offset --eval_type sine_1Hz_10deg_-40offset &
wait
python main.py --ftype 0 --model_dir exp/amp_dependency --data_type sine_1Hz_50deg_0offset --eval_type sine_1Hz_10deg_40offset &
python main.py --ftype 1 --model_dir exp/amp_dependency --data_type sine_1Hz_50deg_0offset --eval_type sine_1Hz_10deg_40offset &
python main.py --ftype 2 --model_dir exp/amp_dependency --data_type sine_1Hz_50deg_0offset --eval_type sine_1Hz_10deg_40offset &
wait

python main.py --ftype 0 --model_dir exp/amp_dependency --data_type sine_0.2Hz_50deg_0offset --eval_type sine_1Hz_20deg_0offset &
python main.py --ftype 1 --model_dir exp/amp_dependency --data_type sine_0.2Hz_50deg_0offset --eval_type sine_1Hz_20deg_0offset &
python main.py --ftype 2 --model_dir exp/amp_dependency --data_type sine_0.2Hz_50deg_0offset --eval_type sine_1Hz_20deg_0offset &
wait
python main.py --ftype 0 --model_dir exp/amp_dependency --data_type sine_0.2Hz_50deg_0offset --eval_type sine_1Hz_10deg_-40offset &
python main.py --ftype 1 --model_dir exp/amp_dependency --data_type sine_0.2Hz_50deg_0offset --eval_type sine_1Hz_10deg_-40offset &
python main.py --ftype 2 --model_dir exp/amp_dependency --data_type sine_0.2Hz_50deg_0offset --eval_type sine_1Hz_10deg_-40offset &
wait
python main.py --ftype 0 --model_dir exp/amp_dependency --data_type sine_0.2Hz_50deg_0offset --eval_type sine_1Hz_10deg_40offset &
python main.py --ftype 1 --model_dir exp/amp_dependency --data_type sine_0.2Hz_50deg_0offset --eval_type sine_1Hz_10deg_40offset &
python main.py --ftype 2 --model_dir exp/amp_dependency --data_type sine_0.2Hz_50deg_0offset --eval_type sine_1Hz_10deg_40offset &
wait

# random_walk
python main.py --ftype 0 --model_dir exp/random_walk --data_type random_walk_30deg_1seed --eval_type random_walk_30deg_2seed &
python main.py --ftype 1 --model_dir exp/random_walk --data_type random_walk_30deg_1seed --eval_type random_walk_30deg_2seed &
python main.py --ftype 2 --model_dir exp/random_walk --data_type random_walk_30deg_1seed --eval_type random_walk_30deg_2seed &
wait