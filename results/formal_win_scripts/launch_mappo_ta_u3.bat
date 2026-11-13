@echo off
cd /d F:\Codes\Python\MultiUAV-v2
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
.venv\Scripts\python.exe experiment\run_exclusive_marl_pilot.py --algorithm MAPPO --uav-num 3 --episodes 3000 --horizon-steps 2000 --eval-interval 50 --seed 42 --guide-prob 0.5 --guide-min-prob 0.5 --teacher-anchor-weight 10.0 --teacher-action-blend 0.5 --actor-lr 1e-4 --critic-lr 1e-3 --safety-outage-threshold 0.2 --routes-json configs\exclusive\u3_s42_4d_ga_routes.json --run-root results\formal_win --run-id win-formal-u3-s42-mappo-ta >> results\formal_win_logs\win-formal-u3-s42-mappo-ta.log 2>&1