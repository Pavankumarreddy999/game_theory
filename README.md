# Dynamic Games in Trading

Frequency-dependent payoffs and bounded (sticky) capital flows, with noisy switching and real-data calibration across four industries.

## Run order
1. `python fetch_data.py` (download data, calibration table)
2. `python run_tournament.py --file_a data/raw/alcohol_Diageo.csv --file_b data/raw/alcohol_AB_InBev.csv`
3. `python simulate_market.py --industry alcohol`
4. `python simulate_evolution.py --industry all`
5. `python sweep_v2.py`, `python phase5.py --industry alcohol`, `python phase6.py`
6. `python make_figures.py`

Setup: `conda create -n tradinggame python=3.11 -y`, `conda activate tradinggame`, `pip install -r requirements.txt`
