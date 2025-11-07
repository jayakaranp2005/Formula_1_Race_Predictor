import pandas as pd
import joblib

def predict_winners(model_path, new_data_path, upcoming_race_id, grand_prix_name, output_path=None):
    # =============================
    # 🏎️ 1. Driver ID → Name Map
    # =============================
    driver_map = {
        1: "VER",   # Max Verstappen
        11: "PER",  # Sergio Pérez
        16: "LEC",  # Charles Leclerc
        55: "SAI",  # Carlos Sainz
        44: "HAM",  # Lewis Hamilton
        63: "RUS",  # George Russell
        4: "NOR",   # Lando Norris
        81: "PIA",  # Oscar Piastri
        10: "GAS",  # Pierre Gasly
        31: "OCO",  # Esteban Ocon
        18: "STR",  # Lance Stroll
        14: "ALO",  # Fernando Alonso
        27: "HUL",  # Nico Hülkenberg
        20: "MAG",  # Kevin Magnussen
        23: "ALB",  # Alex Albon
        2:  "SAR",  # Logan Sargeant
        24: "ZHO",  # Zhou Guanyu
        77: "BOT",  # Valtteri Bottas
        3:  "RIC",  # Daniel Ricciardo
        22: "TSU",  # Yuki Tsunoda
    }

    # =============================
    # 2. Load Model & Data
    # =============================
    model = joblib.load(model_path)
    X_pred_full = pd.read_csv(new_data_path)

    # Keep Driver_ID and drop from features
    X_pred = X_pred_full.drop(columns=['Driver_ID']) if 'Driver_ID' in X_pred_full.columns else X_pred_full.copy()

    # =============================
    # 3. Predict Win Probabilities
    # =============================
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_pred)[:, 1]
    else:
        probs = model.predict(X_pred)

    X_pred_full['Win_Probability'] = probs

    # =============================
    # 4. Select Winners / Top 3
    # =============================
    threshold = 0.5
    winners = X_pred_full[X_pred_full['Win_Probability'] >= threshold]

    if winners.empty:
        winners = X_pred_full.nlargest(3, 'Win_Probability')

    # Extract year for display
    year = upcoming_race_id.split('_')[0] if '_' in upcoming_race_id else upcoming_race_id

    # =============================
    # 5. Print Results
    # =============================
    print(f"\n🏁 Predictions for the {year} {grand_prix_name} 🏁")
    print("🏆 Predicted Winners:")
    for i, row in enumerate(winners.itertuples(), 1):
        driver_id = row.Driver_ID
        driver_name = driver_map.get(driver_id, f"Driver_{driver_id}")  # fallback if unknown
        print(f"{i}. {driver_name} (Driver ID: {driver_id}) — Win Probability: {row.Win_Probability:.3f}")

    # =============================
    # 6. Optional Save
    # =============================
    if output_path:
        X_pred_full.to_csv(output_path, index=False)
        print(f"\n✅ Predictions saved to: {output_path}")


# ==========================================
# Example Run for Mexico City Grand Prix
# ==========================================
if __name__ == "__main__":
    predict_winners(
        model_path='models/final_xgb_model.pkl',
        new_data_path='data/new_data.csv',
        upcoming_race_id='2025_20',
        grand_prix_name='Mexico City Grand Prix',
        output_path='data/predictions.csv'
    )
