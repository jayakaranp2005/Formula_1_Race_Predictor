# Formula_1_Race_Predictor

A powerful Formula 1 race predictor.

This project uses the fastf1 Python API to collect race and session data from 2022 through the current season.

This README explains how to create a virtual environment, install dependencies, the repository layout, provides one-line script descriptions with the exact Python terminal commands to run them, and documents the mandatory CSV required before running the upcoming-data/prediction step.

## Quickstart — create the virtual environment and install requirements

1. Clone the repository:
```bash
git clone https://github.com/jayakaranp2005/Formula_1_Race_Predictor.git
cd Formula_1_Race_Predictor
```

2. Create and activate a virtual environment (Linux / macOS):
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project structure

```text
Formula_1_Race_Predictor/
│
├── data/
│
├── models/
│
├── src/
│   ├── __pycache__/ 
│   ├── __init__.py
│   ├── data_fetcher.py
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── model_trainer_hyperparameter.py
│   ├── final_model_trainer.py
│   ├── final_evaluator.py
│   ├── upcoming_data_fetcher.py
    ├── upcomind_input_helper.py
│   └── FINAL_PREDICTOR.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

Create directories if missing:
```bash
mkdir -p data models
```

## Typical pipeline — single-line descriptions + python terminal commands

- src/data_fetcher.py — fetches historical race and driver raw data.  
  Run:
  ```bash
  python src/data_fetcher.py
  ```

- src/data_preparation.py — cleans, normalizes, and merges raw datasets.  
  Run:
  ```bash
  python src/data_preparation.py
  ```

- src/feature_engineering.py — constructs and encodes model features.  
  Run:
  ```bash
  python src/feature_engineering.py
  ```

- src/model_trainer_hyperparameter.py — performs hyperparameter search for candidate models.  
  Run:
  ```bash
  python src/model_trainer_hyperparameter.py
  ```

- src/final_model_trainer.py — trains final model(s) using chosen hyperparameters.  
  Run:
  ```bash
  python src/final_model_trainer.py
  ```

- src/final_evaluator.py — computes evaluation metrics on holdout/test data.  
  Run:
  ```bash
  python src/final_evaluator.py
  ```

- src/upcoming_data_fetcher.py — reads upcoming race input file(s) and prepares inputs for prediction.  
  Example run (path to your CSV):
  ```bash
  python src/upcoming_data_fetcher.py data/upcoming_qualifying.csv
  ```

- src/FINAL_PREDICTOR.py — loads trained model(s) and produces predictions for the provided inputs.  
  Example run (input CSV and trained model path):
  ```bash
  python src/FINAL_PREDICTOR.py data/upcoming_qualifying.csv models/final_model.pkl
  ```

## Mandatory input before running upcoming_data_fetcher.py

Before executing src/upcoming_data_fetcher.py you must create a CSV file (example name: `upcoming_qualifying.csv`) containing one row per driver for the target race. Required columns and formatting:

- Columns (exact headers required):
  - Driver_ID
  - Grid_Position
  - Qualifying_Time
  - Race_ID
  - Circuit_Name

- Qualifying_Time must be in seconds (numeric, total seconds as a float or int).  
- Circuit_Name must exactly match the circuit names used in training (same spelling and casing used during feature encoding), e.g. Mexico City Grand Prix

Example header and one row:
```csv
Driver_ID,Grid_Position,Qualifying_Time,Race_ID,Circuit_Name
hamilton,1,85.432,2025_Australia,Albert Park
```

Place the CSV in the `data/` folder or pass its path to the scripts as shown above.

## Example output (sample)

When you run the final predictor you may see output formatted like:

🏁 Predictions for the 2025 Mexico City Grand Prix 🏁  
🏆 Predicted Winners:

1. NOR (Driver ID: 4) — Win Probability: 0.636  
2. LEC (Driver ID: 16) — Win Probability: 0.694

## Notes

- Handle large raw data and sensitive files properly.  
- Check each script for CLI arguments or config options at the top of the file and adapt file paths accordingly.

  ## Thank me later :)
