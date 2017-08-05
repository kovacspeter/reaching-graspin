#!/usr/bin/env bash
python TRPO_prevdir.py --scope grasper_prevdir300_1 --n_trajs 300 --checkpoint_freq 100 --env_name Grasper3d-v0 --is_train True --is_load Tru --is_monitor False --n_processes 1 &
python TRPO_prevdir.py --scope grasper_prevdir300_2 --n_trajs 300 --checkpoint_freq 100 --env_name Grasper3d-v0 --is_train True --is_load Tru --is_monitor False --n_processes 1 &
python TRPO_prevdir.py --scope grasper_prevdir300_3 --n_trajs 300 --checkpoint_freq 100 --env_name Grasper3d-v0 --is_train True --is_load Tru --is_monitor False --n_processes 1 &
python TRPO_prevdir.py --scope grasper_prevdir300_4 --n_trajs 300 --checkpoint_freq 100 --env_name Grasper3d-v0 --is_train True --is_load Tru --is_monitor False --n_processes 1 &
python TRPO_prevdir.py --scope grasper_prevdir300_5 --n_trajs 300 --checkpoint_freq 100 --env_name Grasper3d-v0 --is_train True --is_load Tru --is_monitor False --n_processes 1
