import json
from pathlib import Path
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/seaborn not found. Plotting will be skipped.")

def plot_track_a():
    report_path = Path("runs/m6_behavioral_characterization/behavioral_report.json")
    if not report_path.exists():
        return
        
    with open(report_path, "r") as f:
        data = json.load(f)
        
    alphas = sorted([float(a) for a in data.keys()])
    metrics = list(data[str(alphas[0])]["authority"].keys())
    
    if not HAS_MATPLOTLIB:
        print("Track A Report Data available. Install matplotlib to generate plots.")
        return
        
    out_dir = Path("runs/m6_behavioral_characterization/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        auth_means = [data[str(a)]["authority"][metric]["mean"] for a in alphas]
        rand_means = [data[str(a)]["random"][metric]["mean"] for a in alphas]
        
        auth_stds = [data[str(a)]["authority"][metric]["std"] for a in alphas]
        rand_stds = [data[str(a)]["random"][metric]["std"] for a in alphas]
        
        plt.plot(alphas, auth_means, label="Authority Injection", marker='o', color='blue')
        plt.plot(alphas, rand_means, label="Random Injection", marker='s', color='gray')
        
        # Shade std deviations
        plt.fill_between(alphas, 
                         [m - s for m, s in zip(auth_means, auth_stds)], 
                         [m + s for m, s in zip(auth_means, auth_stds)], 
                         alpha=0.2, color='blue')
                         
        plt.fill_between(alphas, 
                         [m - s for m, s in zip(rand_means, rand_stds)], 
                         [m + s for m, s in zip(rand_means, rand_stds)], 
                         alpha=0.2, color='gray')
        
        plt.title(f"Track A: {metric.replace('_', ' ').title()} vs Alpha")
        plt.xlabel("Alpha (Injection Magnitude)")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()
        plt.grid(True)
        
        plt.savefig(out_dir / f"track_a_{metric}.png", dpi=300)
        plt.close()
        
    print(f"Track A Plots saved to {out_dir}")

def plot_track_b():
    report_path = Path("runs/m6_behavioral_consequences/compliance_report.json")
    if not report_path.exists():
        return
        
    with open(report_path, "r") as f:
        data = json.load(f)
        
    alphas = sorted([float(a) for a in data.keys()])
    
    if not HAS_MATPLOTLIB:
        print("Track B Report Data available. Install matplotlib to generate plots.")
        return
        
    out_dir = Path("runs/m6_behavioral_consequences/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    auth_refusal = [data[str(a)]["authority_refusal_rate"] * 100 for a in alphas]
    rand_refusal = [data[str(a)]["random_refusal_rate"] * 100 for a in alphas]
    
    plt.plot(alphas, auth_refusal, label="Authority Injection", marker='o', color='red')
    plt.plot(alphas, rand_refusal, label="Random Injection", marker='s', color='gray')
    
    plt.title("Track B: Refusal Rate on Sensitive Prompts (SENSY) vs Alpha")
    plt.xlabel("Alpha (Injection Magnitude)")
    plt.ylabel("Refusal Rate (%)")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(out_dir / "track_b_refusal_rate.png", dpi=300)
    plt.close()
    
    print(f"Track B Plots saved to {out_dir}")

if __name__ == "__main__":
    plot_track_a()
    plot_track_b()
