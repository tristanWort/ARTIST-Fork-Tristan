import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_error_evolution_from_multiindex_csv(
    actual_csv_path: str,
    regular_csv_path: str,
    output_dir: str,
    font_size: int = 11
):
    """
    Plot evolution of actual and regular alignment errors from wide multiindex CSVs.
    Saves 6 plots: full, actual-only, and valid-only, each in linear and log scale.
    """
    os.makedirs(output_dir, exist_ok=True)

    def load_mean_errors(csv_path: str) -> pd.DataFrame:
        # Load with multi-indexed columns
        df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        # Collapse over heliostats (columns = (heliostat, mode)), keep only mode level
        df.columns = df.columns.droplevel(0)  # keep mode only
        df_means = df.groupby(level=0, axis=1).mean()  # mean over heliostats per mode
        return df_means

    actual = load_mean_errors(actual_csv_path)
    regular = load_mean_errors(regular_csv_path)
    epochs = actual.index.astype(int)

    # Plotting utility
    def plot_curves(curves: dict, title: str, filename: str, log=False):
        plt.figure(figsize=(10, 6))
        for label, (data, color) in curves.items():
            plt.plot(epochs, data, label=label, color=color)
        plt.xlabel("Epoch", fontsize=font_size)
        plt.ylabel("Tracking Error [mrad]", fontsize=font_size)
        plt.title(title, fontsize=font_size)
        if log:
            plt.yscale("log")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}_{'log' if log else 'linear'}.pdf"))
        plt.close()

    # === 6 Plots ===
    plot_curves({
        "Actual Train": (actual["Train"], "red"),
        "Actual Valid": (actual["Valid"], "orange"),
        "Regular Train": (regular["Train"], "blue"),
        "Regular Valid": (regular["Valid"], "cyan"),
    }, "All Tracking Errors (Train & Valid)", "error_evolution_all", log=False)

    plot_curves({
        "Actual Train": (actual["Train"], "red"),
        "Actual Valid": (actual["Valid"], "orange"),
        "Regular Train": (regular["Train"], "blue"),
        "Regular Valid": (regular["Valid"], "cyan"),
    }, "All Tracking Errors (Train & Valid, Log Scale)", "error_evolution_all", log=True)

    plot_curves({
        "Actual Train": (actual["Train"], "red"),
        "Actual Valid": (actual["Valid"], "orange"),
    }, "Actual Tracking Errors (Train & Valid)", "error_evolution_actual", log=False)

    plot_curves({
        "Actual Train": (actual["Train"], "red"),
        "Actual Valid": (actual["Valid"], "orange"),
    }, "Actual Tracking Errors (Train & Valid, Log Scale)", "error_evolution_actual", log=True)

    plot_curves({
        "Actual Valid": (actual["Valid"], "orange"),
        "Regular Valid": (regular["Valid"], "cyan"),
    }, "Validation Comparison: Actual vs. Regular", "error_evolution_valid", log=False)

    plot_curves({
        "Actual Valid": (actual["Valid"], "orange"),
        "Regular Valid": (regular["Valid"], "cyan"),
    }, "Validation Comparison: Actual vs. Regular (Log Scale)", "error_evolution_valid", log=True)

    print(f"✅ Saved 6 tracking error evolution plots to: {output_dir}")


if __name__ == '__main__':
    run = "/dss/dsshome1/05/di38kid/data/results/runs/run_2506302322_20_Heliostats"
    output_dir = f"{run}/plots/error_evolution"
    
    mode = "Test"
    csv_actual_test_errors = f"{run}/logs/ActualAlignmentErrors_mrad/Avg.csv"
    csv_test_errors = f"{run}/logs/AlignmentErrors_mrad/Avg.csv"
    
    plot_error_evolution_from_multiindex_csv(actual_csv_path=csv_actual_test_errors, regular_csv_path=csv_test_errors, output_dir=output_dir)
