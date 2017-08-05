#!/usr/bin/env bash
python TRPO_prevdir.py --scope grasper_prevdir300_6 --n_trajs 300 --checkpoint_freq 100 --env_name Grasper3d-v0 --is_train True --is_load Tru --is_monitor False --n_processes 1 &
python TRPO_prevdir.py --scope grasper_prevdir300_7 --n_trajs 300 --checkpoint_freq 100 --env_name Grasper3d-v0 --is_train True --is_load Tru --is_monitor False --n_processes 1 &
python TRPO_prevdir.py --scope grasper_prevdir300_8 --n_trajs 300 --checkpoint_freq 100 --env_name Grasper3d-v0 --is_train True --is_load Tru --is_monitor False --n_processes 1 &
python TRPO_prevdir.py --scope grasper_prevdir300_9 --n_trajs 300 --checkpoint_freq 100 --env_name Grasper3d-v0 --is_train True --is_load Tru --is_monitor False --n_processes 1 &
python TRPO_prevdir.py --scope grasper_prevdir300_10 --n_trajs 300 --checkpoint_freq 100 --env_name Grasper3d-v0 --is_train True --is_load Tru --is_monitor False --n_processes 1
