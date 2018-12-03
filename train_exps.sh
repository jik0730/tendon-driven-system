#!/bin/sh

# freq_dependency
# python main.py --ftype 0 --simT 10 --model_dir exp/freq_dependency --data_type sine_0.5Hz_10deg_0offset &
# python main.py --ftype 1 --simT 10 --model_dir exp/freq_dependency --data_type sine_0.5Hz_10deg_0offset &
# python main.py --ftype 2 --simT 10 --model_dir exp/freq_dependency --data_type sine_0.5Hz_10deg_0offset &
# wait
# python main.py --ftype 0 --simT 10 --model_dir exp/freq_dependency --data_type sine_5Hz_10deg_0offset &
# python main.py --ftype 1 --simT 10 --model_dir exp/freq_dependency --data_type sine_5Hz_10deg_0offset &
# python main.py --ftype 2 --simT 10 --model_dir exp/freq_dependency --data_type sine_5Hz_10deg_0offset &
# wait

# amp_denpendency
# python main.py --ftype 0 --simT 10 --model_dir exp/amp_dependency --data_type sine_1Hz_10deg_0offset &
# python main.py --ftype 1 --simT 10 --model_dir exp/amp_dependency --data_type sine_1Hz_10deg_0offset &
# python main.py --ftype 2 --simT 10 --model_dir exp/amp_dependency --data_type sine_1Hz_10deg_0offset &
# wait
# python main.py --ftype 0 --simT 10 --model_dir exp/amp_dependency --data_type sine_1Hz_50deg_0offset &
# python main.py --ftype 1 --simT 10 --model_dir exp/amp_dependency --data_type sine_1Hz_50deg_0offset &
# python main.py --ftype 2 --simT 10 --model_dir exp/amp_dependency --data_type sine_1Hz_50deg_0offset &
# wait
# python main.py --ftype 0 --simT 10 --model_dir exp/amp_dependency --data_type sine_0.2Hz_50deg_0offset &
# python main.py --ftype 1 --simT 10 --model_dir exp/amp_dependency --data_type sine_0.2Hz_50deg_0offset &
# python main.py --ftype 2 --simT 10 --model_dir exp/amp_dependency --data_type sine_0.2Hz_50deg_0offset &
# wait

# random_walk
# python main.py --ftype 0 --simT 10 --model_dir exp/random_walk --data_type random_walk_30deg_3seed &
# python main.py --ftype 1 --simT 10 --model_dir exp/random_walk --data_type random_walk_30deg_3seed &
# python main.py --ftype 2 --simT 10 --model_dir exp/random_walk --data_type random_walk_30deg_3seed &
# wait
python main.py --ftype 0 --simT 13.15 --model_dir exp/random_walk --data_type sine_freq_variation &
python main.py --ftype 1 --simT 13.15 --model_dir exp/random_walk --data_type sine_freq_variation &
python main.py --ftype 2 --simT 13.15 --model_dir exp/random_walk --data_type sine_freq_variation &
wait