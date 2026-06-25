import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import zipfile
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress
from fitparse import FitFile
from pydantic import BaseModel, ConfigDict
from typing import List

# ==========================================
# CONSTANTES DU PROFIL (À MODIFIER PAR L'UTILISATEUR)
# ==========================================
POIDS_KG = 56.0
TAILLE_M = 1.61
FC_MAX = 190  # TODO: Remplacer par votre FC Max réelle

# ==========================================
# CONFIGURATION
# ==========================================
DIR_DERNIER_FIT = "dernier_fit"
DIR_ARCHIVES_FIT = "archives_fit"
FILE_HISTORIQUE = "historique_running.json"

RETENTION_RULES = {
    "Endurance": 10,
    "Fractionné": 10,
    "Seuil": 10,
    "Sortie Longue": 5,
    "Inconnu": 5
}

class RunMetrics(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    date: str
    type: str = ""
    distance_km: float
    duration_min: float
    elevation_gain_m: float
    avg_speed_kmh: float
    avg_cadence_ppm: float
    avg_hr_bpm: float
    max_hr_bpm: float
    time_z1_min: float
    time_z2_min: float
    time_z3_min: float
    trimp: float
    avg_stride_length_m: float
    speed_flat_kmh: float
    speed_hill_kmh: float
    cardiac_drift_bpm_per_min: float
    power_cardio_ratio: float
    efficiency_factor: float
    metabolic_yield_vo2_kg: float
    final_resilience_index: float

def extract_latest_zip():
    if not os.path.exists(DIR_DERNIER_FIT):
        os.makedirs(DIR_DERNIER_FIT)
    
    files = [f for f in os.listdir(DIR_DERNIER_FIT) if f.endswith('.zip')]
    if not files:
        print("Aucun fichier .zip trouvé dans 'dernier_fit'.")
        return None, None
        
    zip_filename = files[0]
    zip_path = os.path.join(DIR_DERNIER_FIT, zip_filename)
    
    extracted_fit_path = None
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.fit'):
                extracted_fit_path = zip_ref.extract(file_info, DIR_DERNIER_FIT)
                break
                
    return zip_path, extracted_fit_path

def parse_fit_to_df(fit_path):
    fitfile = FitFile(fit_path)
    
    # Optimisation mémoire: compréhension de liste
    records = [{data.name: data.value for data in record} for record in fitfile.get_messages('record')]
        
    df = pd.DataFrame(records)
    
    required_cols = ['timestamp', 'position_lat', 'position_long', 'distance', 'enhanced_altitude', 'enhanced_speed', 'heart_rate', 'cadence']
    for col in required_cols:
        if col not in df.columns:
            if col == 'enhanced_speed' and 'speed' in df.columns:
                df['enhanced_speed'] = df['speed']
            elif col == 'enhanced_altitude' and 'altitude' in df.columns:
                df['enhanced_altitude'] = df['altitude']
            else:
                df[col] = np.nan
                
    return df

def clean_data(df):
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    cols_to_interpolate = ['distance', 'enhanced_altitude', 'enhanced_speed', 'heart_rate', 'cadence']
    for col in cols_to_interpolate:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Indexation et rééchantillonnage temporel strict
    df.set_index('timestamp', inplace=True)
    df = df.resample('1s').interpolate(method='time')
    df.reset_index(inplace=True)
    
    window_length = 5
    if len(df) >= window_length:
        # Lissage mathématiquement valide
        df['speed_smoothed'] = savgol_filter(df['enhanced_speed'].fillna(0), window_length, 2)
        df['altitude_smoothed'] = savgol_filter(df['enhanced_altitude'].fillna(0), window_length, 2)
        df['hr_smoothed'] = savgol_filter(df['heart_rate'].fillna(0), window_length, 2)
    else:
        df['speed_smoothed'] = df['enhanced_speed']
        df['altitude_smoothed'] = df['enhanced_altitude']
        df['hr_smoothed'] = df['heart_rate']
        
    df['speed_smoothed'] = df['speed_smoothed'].clip(lower=0)
    df['hr_smoothed'] = df['hr_smoothed'].clip(lower=0)
    
    # Pente sécurisée (somme glissante sur 10s)
    dist_10s = df['distance'].diff().rolling(10).sum()
    alt_10s = df['altitude_smoothed'].diff().rolling(10).sum()
    df['slope_pct'] = np.where(dist_10s > 2.0, (alt_10s / dist_10s) * 100, 0)
    
    df['delta_alt'] = df['altitude_smoothed'].diff().fillna(0)
    
    return df

def compute_metrics(df) -> RunMetrics:
    total_distance_km = df['distance'].max() / 1000 if not df['distance'].isna().all() else 0
    duration_s = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() if not df.empty else 0
    duration_min = duration_s / 60
    
    elevation_gain_m = df.loc[df['delta_alt'] > 0, 'delta_alt'].sum()
    
    avg_speed_ms = df['speed_smoothed'].mean()
    avg_speed_kmh = avg_speed_ms * 3.6
    
    cadence_mean = df.loc[df['cadence'] > 0, 'cadence'].mean()
    avg_cadence_ppm = cadence_mean * 2 if cadence_mean < 120 else cadence_mean
    
    hr_mean = df['hr_smoothed'].mean()
    max_hr_bpm = df['hr_smoothed'].max()
    
    z1_mask = df['hr_smoothed'] < 0.75 * FC_MAX
    z2_mask = (df['hr_smoothed'] >= 0.75 * FC_MAX) & (df['hr_smoothed'] < 0.88 * FC_MAX)
    z3_mask = df['hr_smoothed'] >= 0.88 * FC_MAX
    
    time_z1_min = z1_mask.sum() / 60
    time_z2_min = z2_mask.sum() / 60
    time_z3_min = z3_mask.sum() / 60
    
    ratio_fc = hr_mean / FC_MAX
    trimp = duration_min * ratio_fc * 0.64 * np.exp(1.92 * ratio_fc)
    
    avg_stride_length_m = avg_speed_ms / (avg_cadence_ppm / 60) if avg_cadence_ppm > 0 else 0
    
    flat_mask = df['slope_pct'].abs() < 1.0
    hill_mask = df['slope_pct'] > 3.0
    speed_flat = df.loc[flat_mask, 'speed_smoothed'].mean() * 3.6
    speed_hill = df.loc[hill_mask, 'speed_smoothed'].mean() * 3.6
    
    half_time = duration_s / 2
    stable_mask = flat_mask & (df['speed_smoothed'] > 2.22) & (df['timestamp'] > df['timestamp'].min() + pd.Timedelta(seconds=half_time))
    stable_df = df[stable_mask]
    
    if len(stable_df) > 60:
        x = (stable_df['timestamp'] - stable_df['timestamp'].min()).dt.total_seconds() / 60
        y = stable_df['hr_smoothed']
        slope, intercept, r, p, se = linregress(x, y)
        cardiac_drift_bpm_per_min = slope
    else:
        cardiac_drift_bpm_per_min = 0.0
        
    power_cardio_ratio = (avg_speed_ms / hr_mean) * 100 if hr_mean > 0 else 0
    
    v_m_min = avg_speed_ms * 60
    efficiency_factor = v_m_min / hr_mean if hr_mean > 0 else 0
    
    avg_slope_frac = df.loc[df['delta_alt'] > 0, 'slope_pct'].mean() / 100
    if pd.isna(avg_slope_frac): avg_slope_frac = 0
    vo2_est = 3.5 + 0.2 * v_m_min + 0.9 * v_m_min * avg_slope_frac
    
    last_10_pct_start = df['timestamp'].min() + pd.Timedelta(seconds=duration_s * 0.9)
    last_10_df = df[df['timestamp'] >= last_10_pct_start]
    if not last_10_df.empty and avg_speed_ms > 0:
        final_resilience_index = (last_10_df['speed_smoothed'].mean() / avg_speed_ms) * 100
    else:
        final_resilience_index = 100.0
        
    date_str = df['timestamp'].min().strftime('%Y-%m-%d %H:%M')
    
    metrics = RunMetrics(
        date=date_str,
        distance_km=round(total_distance_km, 2),
        duration_min=round(duration_min, 2),
        elevation_gain_m=round(elevation_gain_m, 0),
        avg_speed_kmh=round(avg_speed_kmh, 2),
        avg_cadence_ppm=round(avg_cadence_ppm, 0) if not pd.isna(avg_cadence_ppm) else 0.0,
        avg_hr_bpm=round(hr_mean, 0),
        max_hr_bpm=round(max_hr_bpm, 0),
        time_z1_min=round(time_z1_min, 1),
        time_z2_min=round(time_z2_min, 1),
        time_z3_min=round(time_z3_min, 1),
        trimp=round(trimp, 1),
        avg_stride_length_m=round(avg_stride_length_m, 2),
        speed_flat_kmh=round(speed_flat, 2) if not pd.isna(speed_flat) else 0.0,
        speed_hill_kmh=round(speed_hill, 2) if not pd.isna(speed_hill) else 0.0,
        cardiac_drift_bpm_per_min=round(cardiac_drift_bpm_per_min, 2),
        power_cardio_ratio=round(power_cardio_ratio, 2),
        efficiency_factor=round(efficiency_factor, 2),
        metabolic_yield_vo2_kg=round(vo2_est, 1),
        final_resilience_index=round(final_resilience_index, 1)
    )
    return metrics

def classify_session(metrics: RunMetrics) -> str:
    if metrics.duration_min > 75 and metrics.avg_hr_bpm < 0.8 * FC_MAX:
        return "Sortie Longue"
    elif metrics.avg_hr_bpm < 0.75 * FC_MAX and metrics.time_z3_min < 5:
        return "Endurance"
    elif metrics.time_z3_min >= 5 or metrics.max_hr_bpm > 0.9 * FC_MAX:
        return "Fractionné"
    elif 0.80 * FC_MAX <= metrics.avg_hr_bpm <= 0.88 * FC_MAX:
        return "Seuil"
    return "Endurance"

def update_history(metrics: RunMetrics, session_type: str):
    metrics.type = session_type
    
    history = {}
    if os.path.exists(FILE_HISTORIQUE):
        try:
            with open(FILE_HISTORIQUE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except json.JSONDecodeError:
            pass
            
    if session_type not in history:
        history[session_type] = []
        
    # Serialize pydantic model to dict
    session_data = metrics.model_dump()
    history[session_type].append(session_data)
    
    history[session_type].sort(key=lambda x: x['date'], reverse=True)
    
    limit = RETENTION_RULES.get(session_type, 5)
    history[session_type] = history[session_type][:limit]
    
    with open(FILE_HISTORIQUE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
        
    return history

def print_analysis(metrics: RunMetrics, session_type: str, history: dict):
    print("\n" + "="*50)
    print("🏃 ANALYSE DE LA SÉANCE - PROFIL INGÉNIERIE")
    print("="*50)
    print(f"📅 Date: {metrics.date}")
    print(f"🏷️ Type identifié : {session_type.upper()}")
    print("-" * 50)
    
    print("\n📊 MACRO INDICATEURS :")
    print(f"  • Distance : {metrics.distance_km} km | Dénivelé: +{metrics.elevation_gain_m} m")
    print(f"  • Durée : {metrics.duration_min} min")
    print(f"  • Vitesse Moyenne : {metrics.avg_speed_kmh} km/h (Plat: {metrics.speed_flat_kmh} / Côte: {metrics.speed_hill_kmh})")
    print(f"  • Fréquence Cardiaque : {metrics.avg_hr_bpm} bpm (Max: {metrics.max_hr_bpm})")
    print(f"  • TRIMP (Charge) : {metrics.trimp}")
    
    print("\n⚙️ BIOMÉCANIQUE & RENDEMENT :")
    print(f"  • Cadence Moyenne : {metrics.avg_cadence_ppm} ppm")
    print(f"  • Foulée Moyenne : {metrics.avg_stride_length_m} m")
    print(f"  • Ratio Puissance/Cardio : {metrics.power_cardio_ratio} (Indice d'efficience)")
    print(f"  • Efficiency Factor (Couplage) : {metrics.efficiency_factor} m/bt")
    print(f"  • Coût O2 (Estimé) : {metrics.metabolic_yield_vo2_kg} ml/min/kg")
    
    print("\n🩺 MÉTRIQUES ÉVOLUTIVES :")
    drift_symbol = "⚠️ Dérive positive (Fatigue/Chaleur)" if metrics.cardiac_drift_bpm_per_min > 0.5 else "✅ Stable"
    print(f"  • Dérive Cardiaque : {metrics.cardiac_drift_bpm_per_min} bpm/min [{drift_symbol}]")
    resilience = "✅ Fort finish" if metrics.final_resilience_index >= 100 else "⚠️ Baisse de régime"
    print(f"  • Indice de Résilience (Fin de course) : {metrics.final_resilience_index}% [{resilience}]")
    print(f"  • Temps en Zones : Z1={metrics.time_z1_min}m | Z2={metrics.time_z2_min}m | Z3={metrics.time_z3_min}m")
    
    past_sessions = [s for s in history.get(session_type, []) if s['date'] != metrics.date]
    if past_sessions:
        print("\n📈 COMPARAISON HISTORIQUE (Même type de séance) :")
        avg_past_speed = round(sum(s.get('avg_speed_kmh', 0) for s in past_sessions) / len(past_sessions), 2)
        avg_past_hr = round(sum(s.get('avg_hr_bpm', 0) for s in past_sessions) / len(past_sessions), 0)
        
        speed_diff = round(metrics.avg_speed_kmh - avg_past_speed, 2)
        hr_diff = metrics.avg_hr_bpm - avg_past_hr
        
        print(f"  • Vitesse : {metrics.avg_speed_kmh} km/h vs {avg_past_speed} km/h (Diff: {speed_diff:+.2f})")
        print(f"  • Cardio : {metrics.avg_hr_bpm} bpm vs {avg_past_hr} bpm (Diff: {hr_diff:+.0f})")
        
        if speed_diff > 0 and hr_diff <= 0:
            print("  💡 CONCLUSION : Excellent! Meilleure vitesse pour un effort cardiaque égal ou inférieur. Gain d'efficience.")
        elif speed_diff > 0 and hr_diff > 0:
            print("  💡 CONCLUSION : Plus rapide mais plus coûteux cardiologiquement. Normal si l'objectif était l'intensité.")
        elif speed_diff < 0 and hr_diff > 0:
            print("  💡 CONCLUSION : Vigilance. Vitesse en baisse et cardio en hausse. Signe possible de fatigue ou surentraînement.")
    print("="*50 + "\n")

def main():
    print("Démarrage de l'analyse...")
    zip_path, extracted_fit = extract_latest_zip()
    
    if not extracted_fit:
        return
        
    print(f"Fichier extrait : {extracted_fit}")
    
    df = parse_fit_to_df(extracted_fit)
    print(f"Données brutes chargées : {len(df)} points.")
    
    df_clean = clean_data(df)
    print("Nettoyage et lissage terminés.")
    
    metrics = compute_metrics(df_clean)
    
    session_type = classify_session(metrics)
    history = update_history(metrics, session_type)
    print("Historique mis à jour.")
    
    print_analysis(metrics, session_type, history)
    
    if not os.path.exists(DIR_ARCHIVES_FIT):
        os.makedirs(DIR_ARCHIVES_FIT)
    
    archive_path = os.path.join(DIR_ARCHIVES_FIT, os.path.basename(zip_path))
    shutil.move(zip_path, archive_path)
    print(f"Fichier archivé dans : {archive_path}")
    
    if os.path.exists(extracted_fit):
        os.remove(extracted_fit)
        
    print("Processus terminé avec succès.")

if __name__ == "__main__":
    main()
