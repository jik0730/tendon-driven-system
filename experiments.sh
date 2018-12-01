#!/bin/sh

python main.py --ftype 0 --simT 10 --model_dir exp/freq_dependency --data_type sine_0.5Hz_10deg
python main.py --ftype 1 --simT 10 --model_dir exp/freq_dependency --data_type sine_0.5Hz_10deg
python main.py --ftype 2 --simT 10 --model_dir exp/freq_dependency --data_type sine_0.5Hz_10deg

python main.py --ftype 0 --simT 10 --model_dir exp/freq_dependency --data_type sine_10Hz_10deg
python main.py --ftype 1 --simT 10 --model_dir exp/freq_dependency --data_type sine_10Hz_10deg
python main.py --ftype 2 --simT 10 --model_dir exp/freq_dependency --data_type sine_10Hz_10deg