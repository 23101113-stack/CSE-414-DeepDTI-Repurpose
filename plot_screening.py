import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_screening_results():
    results_path = os.path.join("results", "repurposing_screening_results.csv")
    if not os.path.exists(results_path):
        print("Screening results CSV not found!")
        return

    df = pd.read_csv(results_path)

    # Sort descending
    df = df.sort_values(by="Predicted_pKd", ascending=True)

    plt.figure(figsize=(8, 4.5))
    palette = sns.color_palette("mako", len(df))
    
    bars = plt.barh(df["Drug_Name"], df["Predicted_pKd"], color=palette, edgecolor='black', linewidth=0.6)
    
    # Value labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.08, bar.get_y() + bar.get_height() / 2,
                 f"{width:.2f}", ha='left', va='center', fontsize=10, fontweight='bold')

    plt.title("Virtual Screening: Prioritized Candidates for Dengue NS5 Target", fontsize=12, pad=12)
    plt.xlabel("Predicted Binding Affinity ($pK_d$)", fontsize=11)
    plt.ylabel("Candidate Compound", fontsize=11)
    plt.xlim(0, max(df["Predicted_pKd"]) + 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_img = os.path.join("results", "dengue_screening_ranked.png")
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"Ranking bar plot saved at: {output_img}")

if __name__ == "__main__":
    plot_screening_results()