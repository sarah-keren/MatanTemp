import argparse
from datetime import datetime
from typing import Sequence
import yaml
import matplotlib.pyplot as plt


def load_demand_values(config_path: str) -> Sequence[tuple[datetime, float]]:
    """Load demand values from YAML config file.

    Returns a list of (time, value) tuples sorted by time.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    values = data["demand_data"]["values"]
    parsed = []
    for time_str, value in values.items():
        time_obj = datetime.strptime(str(time_str), "%H:%M")
        parsed.append((time_obj, float(value)))

    parsed.sort(key=lambda x: x[0])
    return parsed


def plot_demand_pattern(pairs: Sequence[tuple[datetime, float]], output_path: str) -> None:
    times, demands = zip(*pairs)
    plt.figure(figsize=(12, 6))
    plt.plot(times, demands, marker="o")
    plt.title("Demand Pattern")
    plt.xlabel("Time")
    plt.ylabel("Demand")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot demand pattern from YAML configuration")
    parser.add_argument("config", nargs="?", default="configs/demand_data_sample.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--output", default="demand_plot.png", help="Output image path")
    args = parser.parse_args()

    pairs = load_demand_values(args.config)
    plot_demand_pattern(pairs, args.output)