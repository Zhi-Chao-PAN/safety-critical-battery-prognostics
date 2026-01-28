import arviz as az
from pathlib import Path

POOLED_TRACE = Path("results/bayes/trace_pooled.nc")
HIER_TRACE = Path("results/bayes/trace_hierarchical.nc")

def main():
    print("Loading traces...")

    if not POOLED_TRACE.exists():
        raise FileNotFoundError(f"Missing pooled trace: {POOLED_TRACE}")

    if not HIER_TRACE.exists():
        raise FileNotFoundError(f"Missing hierarchical trace: {HIER_TRACE}")

    idata_pool = az.from_netcdf(POOLED_TRACE)
    idata_hier = az.from_netcdf(HIER_TRACE)

    print("Model comparison (WAIC / LOO):")

    comp = az.compare(
        {
            "pooled": idata_pool,
            "hierarchical": idata_hier
        },
        ic="loo"
    )

    print(comp)


if __name__ == "__main__":
    main()
