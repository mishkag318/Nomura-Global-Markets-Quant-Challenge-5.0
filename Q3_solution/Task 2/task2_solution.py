from typing import List
import pandas as pd


DATA_FILE = "trade_data.csv"
DEFAULT_TAU = [5, 10, 15, 20, 25, 30]


def load_data():
    return pd.read_csv(DATA_FILE)


def expected_pnl(client: str, tau: List[int]) -> dict:
    """
    Parameters:
        client: Client identifier
        tau: List of horizons e.g. [5, 10, 15, 20, 25, 30]

    Returns:
        Dictionary with:
        'per_horizon': expected PnL per trade at each tau
        'aggregate': expected aggregate PnL per trade
    """
    data = load_data()
    client_data = data[data["Name"] == client]

    per_horizon = []

    for t in tau:
        price_col = f"M{t}"

        pnl_values = (
            client_data["Side"]
            * client_data["Volume"]
            * (client_data[price_col] - client_data["Trade Price"])
        )

        per_horizon.append(float(pnl_values.mean()))

    aggregate = float(sum(per_horizon) / len(per_horizon))

    return {
        "per_horizon": per_horizon,
        "aggregate": aggregate
    }


def classify_client(client: str) -> str:
    """
    Returns:
        'profitable' or 'costly'
        Based on aggregate expected PnL.
    """
    result = expected_pnl(client, DEFAULT_TAU)

    if result["aggregate"] >= 0:
        return "profitable"
    else:
        return "costly"


def min_half_spread(client: str) -> float:
    """
    Returns:
        Minimum half-spread in price units such that expected aggregate PnL >= 0
        if the LP quotes at M0 ± delta_star.
    """
    data = load_data()
    client_data = data[data["Name"] == client]

    no_spread_per_horizon = []

    for t in DEFAULT_TAU:
        price_col = f"M{t}"

        pnl_values = (
            client_data["Side"]
            * client_data["Volume"]
            * (client_data[price_col] - client_data["M0"])
        )

        no_spread_per_horizon.append(float(pnl_values.mean()))

    no_spread_aggregate = sum(no_spread_per_horizon) / len(no_spread_per_horizon)

    average_volume = client_data["Volume"].mean()

    delta_star = max(0, -no_spread_aggregate / average_volume)

    return float(delta_star)


def make_results_csv():
    data = load_data()
    clients = sorted(data["Name"].unique())

    rows = []

    for client in clients:
        result = expected_pnl(client, DEFAULT_TAU)
        per_horizon = result["per_horizon"]
        aggregate = result["aggregate"]
        delta_star = min_half_spread(client)

        row = {
            "client": client,
            "tau=5": per_horizon[0],
            "tau=10": per_horizon[1],
            "tau=15": per_horizon[2],
            "tau=20": per_horizon[3],
            "tau=25": per_horizon[4],
            "tau=30": per_horizon[5],
            "agg_pnl": aggregate,
            "delta_star": delta_star
        }

        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv("task2_results.csv", index=False)

    print(results)


if __name__ == "__main__":
    make_results_csv()