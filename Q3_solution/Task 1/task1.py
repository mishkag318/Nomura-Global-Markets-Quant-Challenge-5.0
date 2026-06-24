import pandas as pd
from typing import List

df = pd.read_csv("trade_data.csv")

import pandas as pd
from typing import List

df = pd.read_csv("trade_data.csv")

def adversity_profile(client: str, tau: List[int]) -> List[float]:
    client_trades = df[df["Name"] == client]
    percentages = []

    for t in tau:
        price_after_t = client_trades[f"M{t}"]
        pnl_sign = client_trades["Side"] * (price_after_t - client_trades["Trade Price"])

        adverse_count = 0
        #counting number of adverse trades by client
        for value in pnl_sign:
            if value < 0:
                adverse_count += 1
        #adverse/total trades into 100= percentage required
        adverse_percentage = (adverse_count / len(pnl_sign)) * 100
        percentages.append(adverse_percentage)

    return percentages


if __name__ == "__main__":
    tau_values = [5, 10, 15, 20, 25, 30]
    clients = sorted(df["Name"].unique())

    rows = []

    for client in clients:
        profile = adversity_profile(client, tau_values)
        row = [client] + profile
        rows.append(row)

    result_df = pd.DataFrame(
        rows,
        columns=["client"] + [f"tau={t}" for t in tau_values]
    )

    result_df.to_csv("task1_results.csv", index=False)

    print(result_df)
    print("Saved task1_results.csv")