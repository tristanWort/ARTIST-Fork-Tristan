import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


def plot_loss_evolution(csv_path: str, output_dir: str):
    # Read CSV with multi-level columns
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)

    # Get all available loss types (level 0)
    loss_types = df.columns.levels[0]
    os.makedirs(output_dir, exist_ok=True)

    for loss_type in loss_types:
        # Skip if no Train/Valid pair available
        if not {"Train", "Valid"}.issubset(df[loss_type].columns):
            continue

        df_loss = df[loss_type][["Train", "Valid"]]

        # --- Linear scale plot ---
        plt.figure(figsize=(10, 6))
        plt.plot(df_loss.index, df_loss["Train"], label="Training Loss", color="blue")
        plt.plot(df_loss.index, df_loss["Valid"], label="Validation Loss", color="orange")
        plt.title(f"{loss_type} Evolution over Epochs (Linear Scale)", fontsize=14)
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{loss_type}_linear.pdf"), format="pdf")
        plt.close()

        # --- Log scale plot ---
        plt.figure(figsize=(10, 6))
        plt.plot(df_loss.index, df_loss["Train"], label="Training Loss", color="blue")
        plt.plot(df_loss.index, df_loss["Valid"], label="Validation Loss", color="orange")
        plt.yscale("log")
        plt.title(f"{loss_type} Evolution over Epochs (Log Scale)")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss (log scale)", fontsize=11)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{loss_type}_log.pdf"), format="pdf")
        plt.close()

        print(f"✅ Saved plots for: {loss_type}")


def plot_combined_loss_evolution(csv_path: str, output_dir: str):
    """
    Plot all loss types (Train & Valid) in a single combined figure,
    once with linear scale, once with logarithmic scale.
    """
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    loss_types = df.columns.levels[0]
    os.makedirs(output_dir, exist_ok=True)

    color_cycle = plt.cm.tab10.colors  # 10 distinct colors

    # --- Linear scale ---
    plt.figure(figsize=(12, 7))
    for i, loss_type in enumerate(loss_types):
        if {"Train", "Valid"}.issubset(df[loss_type].columns):
            train_color = color_cycle[i % len(color_cycle)]
            # valid_color = color_cycle[(i + len(loss_types)) % len(color_cycle)]
            plt.plot(df.index, df[loss_type]["Train"], label=f"{loss_type} (Train)", color=train_color)
            plt.plot(df.index, df[loss_type]["Valid"], label=f"{loss_type} (Valid)", color=train_color, linestyle='--')

    plt.title("All Losses Over Epochs (Linear Scale)", fontsize=14)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_losses_linear.pdf"), format="pdf")
    plt.close()

    # --- Log scale ---
    plt.figure(figsize=(12, 7))
    for i, loss_type in enumerate(loss_types):
        if {"Train", "Valid"}.issubset(df[loss_type].columns):
            train_color = color_cycle[i % len(color_cycle)]
            valid_color = color_cycle[(i + len(loss_types)) % len(color_cycle)]
            plt.plot(df.index, df[loss_type]["Train"], label=f"{loss_type} (Train)", color=train_color)
            plt.plot(df.index, df[loss_type]["Valid"], label=f"{loss_type} (Valid)", color=valid_color, linestyle='--')

    plt.yscale("log")
    plt.title("All Losses Over Epochs (Logarithmic Scale)", fontsize=11)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (log scale)", fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_losses_log.pdf"), format="pdf")
    plt.close()

    print("✅ Saved combined loss plots (linear & log)")


# Example usage
if __name__ == "__main__":
    run_dir = Path('/dss/dsshome1/05/di38kid/data/results/runs/run_2506301701_20_Heliostats')
    plot_loss_evolution(run_dir / "logs/losses.csv", run_dir / "plots/loss")
    plot_combined_loss_evolution(run_dir / "logs/losses.csv", run_dir / "plots/loss")
    