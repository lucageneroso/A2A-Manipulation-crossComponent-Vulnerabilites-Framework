"""
analyze_results.py — Generazione grafici e statistiche dalla campagna.
"""

import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Aggiunto per evitare errori Tkinter su Windows
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact

def setup_publication_style():
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    sns.set_theme(style="whitegrid", palette="muted")

def wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (0.0, 1.0)
    from scipy.stats import norm
    import numpy as np
    z = norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

def main():
    # Trova il file CSV più recente
    csv_files = glob.glob("results/campaign_results_*.csv")
    if not csv_files:
        print("Nessun risultato trovato in results/")
        return
        
    latest_csv = max(csv_files, key=os.path.getctime)
    print(f"Analisi del file: {latest_csv}")
    
    df = pd.read_csv(latest_csv)
    
    # Rimuovi trial falliti (es. timeout)
    valid_df = df[df["error"].isna()]
    
    summary = []
    for config, group in valid_df.groupby("config_name"):
        n = len(group)
        successes = group["success"].sum()
        disconnects = group["action_reasoning_disconnect"].sum()
        
        rate = successes / n if n > 0 else 0
        disc_rate = disconnects / n if n > 0 else 0
        
        ci = wilson_ci(successes, n)
        
        summary.append({
            "Config": config,
            "Model": group["model_name"].iloc[0],
            "N": n,
            "Successes": successes,
            "Success Rate": rate,
            "CI Lower": ci[0],
            "CI Upper": ci[1],
            "Disconnects": disconnects,
            "Disconnect Rate": disc_rate
        })
        
    summary_df = pd.DataFrame(summary)
    print("\n--- SOMMARIO ---")
    print(summary_df.to_string(index=False))
    
    # Genera grafico Success Rate
    setup_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(summary_df))
    
    rates = summary_df["Success Rate"]
    errors = [
        rates - summary_df["CI Lower"],
        summary_df["CI Upper"] - rates
    ]
    
    bars = ax.bar(x, rates, yerr=errors, capsize=5, color="skyblue", edgecolor="black")
    
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Config"], rotation=45, ha="right")
    ax.set_ylabel("Attack Success Rate (w/ 95% CI)")
    ax.set_title("Cross-Component Vulnerability by Configuration")
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/success_rates.png")
    print("\nGrafico salvato in results/success_rates.png")
    
if __name__ == "__main__":
    main()
