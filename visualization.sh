#/bin/sh

# python visualization.py exp/freq_dependency/sine_0.5Hz_10deg_0offset/f_est=0/training 10 &
# python visualization.py exp/freq_dependency/sine_0.5Hz_10deg_0offset/MLP/training 10 &
# python visualization.py exp/freq_dependency/sine_0.5Hz_10deg_0offset/oracle/training 10 &
# wait

# python visualization.py exp/random_walk/sine_freq_variation/f_est=0/training 13 &
# python visualization.py exp/random_walk/sine_freq_variation/MLP/training 13 &
# python visualization.py exp/random_walk/sine_freq_variation/oracle/training 13 &
# wait

# python visualization.py exp/random_walk/random_walk_30deg_3seed/f_est=0/training 10 &
# python visualization.py exp/random_walk/random_walk_30deg_3seed/MLP/training 10 &
# python visualization.py exp/random_walk/random_walk_30deg_3seed/oracle/training 10 &
# wait

# python visualization.py exp/amp_dependency/sine_1Hz_50deg_0offset/f_est=0/training 10 &
# python visualization.py exp/amp_dependency/sine_1Hz_50deg_0offset/MLP/training 10 &
# python visualization.py exp/amp_dependency/sine_1Hz_50deg_0offset/oracle/training 10 &
# wait

# python visualization.py exp/random_walk/sine_freq_variation_with_step/f_est=0/training 17 &
# python visualization.py exp/random_walk/sine_freq_variation_with_step/MLP/training 17 &
# python visualization.py exp/random_walk/sine_freq_variation_with_step/oracle/training 17 &
# wait

# python visualization.py exp/random_walk/sine_freq_variation_with_step 17 &
python visualization.py exp/random_walk/random_walk_30deg_3seed 10 &
wait