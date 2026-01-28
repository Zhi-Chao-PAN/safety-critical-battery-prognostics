import pandas as pd
import matplotlib.pyplot as plt

def main():
    curve_path = "results/income_nonlinear_curve.csv"
    df = pd.read_csv(curve_path)

    df = df.sort_values("median_income")

    plt.figure(figsize=(8, 5))
    plt.plot(df["median_income"], df["predicted_price"], label="Nonlinear income-only model")
    plt.xlabel("Median Income")
    plt.ylabel("Predicted Price")
    plt.title("Nonlinear Relationship: Income vs Housing Price")
    plt.grid(True)
    plt.legend()

    save_path = "results/fig_income_nonlinearity.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"Saved figure to: {save_path}")

if __name__ == "__main__":
    main()
