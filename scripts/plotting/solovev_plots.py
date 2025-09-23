from mrx.InputOutput import parse_args
from mrx.Plotting import generate_solovev_plots

if __name__ == "__main__":
    # Get user input
    params = parse_args()
    name = params["name"]

    generate_solovev_plots(name)

    print(f"Plots saved to script_outputs/solovev/{name}/")
