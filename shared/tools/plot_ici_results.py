import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_ici_distribution():
    bins = ['Bin 1\n(0-0.2)\nIsolated', 'Bin 2\n(0.2-0.5)\nMild Overlap', 'Bin 3\n(0.5-1.0)\nHeavy Crowding', 'Bin 4\n(>1.0)\nExtreme Adhesion']
    counts = [351, 532, 696, 173]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bins, counts, color=['#4daf4a', '#377eb8', '#ff7f00', '#e41a1c'], edgecolor='black', alpha=0.8)
    
    plt.title('FaroPigSeg Test Set: Instance Crowding Index (ICI) Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Instance Crowding Index (ICI) Bins', fontsize=12)
    plt.ylabel('Number of Instances', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 10, f"{yval} ({(yval/1752)*100:.1f}%)", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    out_dir = Path('C:/Users/Administrator/.gemini/antigravity/brain/08659641-44b7-4544-8b51-2274a96336ad/images')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'ici_distribution.png'
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300)
    print(f"Saved {out_path}")

if __name__ == '__main__':
    plot_ici_distribution()
