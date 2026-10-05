conda run -n pytorch python tools/run_ablations.py --weight 0.1 --margin 0.1
conda run -n pytorch python tools/run_ablations.py --weight 1.0 --margin 0.1
conda run -n pytorch python tools/run_ablations.py --weight 0.5 --margin 0.3
