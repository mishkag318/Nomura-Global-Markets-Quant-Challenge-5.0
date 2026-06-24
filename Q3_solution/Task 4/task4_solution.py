import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


HORIZONS = [5, 10, 15, 20, 25, 30]

MODELS_BY_TAU = {}
ARTIFACTS = {}


# ============================================================
# TASK 3 MODEL PIPELINE
# ============================================================

def load_data(path="trade_data.csv"):
    return pd.read_csv(path)


def add_trade_features(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["time"].astype(str),
        format="%d-%m-%Y %H:%M:%S",
        errors="raise"
    )

    df = df.sort_values(["timestamp"]).reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["second"] = df["timestamp"].dt.second
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df["seconds_from_midnight"] = (
        df["hour"] * 3600 + df["minute"] * 60 + df["second"]
    )

    df["price_vs_mid"] = df["Trade Price"] - df["M0"]
    df["abs_price_vs_mid"] = df["price_vs_mid"].abs()

    safe_spread = df["Spread"].replace(0, np.nan)

    df["relative_spread"] = df["Spread"] / df["M0"]
    df["price_vs_mid_in_spreads"] = df["price_vs_mid"] / safe_spread
    df["abs_price_vs_mid_in_spreads"] = df["price_vs_mid_in_spreads"].abs()

    df["signed_volume"] = df["Side"] * df["Volume"]
    df["log_volume"] = np.log1p(df["Volume"])

    df["side_x_price_vs_mid"] = df["Side"] * df["price_vs_mid"]
    df["side_x_price_vs_mid_in_spreads"] = (
        df["Side"] * df["price_vs_mid_in_spreads"]
    )
    df["side_x_volume"] = df["Side"] * df["log_volume"]

    # Past-looking market movement features.
    # These use current M0 and older M0 values from the same date.
    for lag in [10, 20, 50]:
        df[f"m0_change_{lag}"] = df["M0"] - df.groupby("Date")["M0"].shift(lag)
        df[f"side_m0_change_{lag}"] = df["Side"] * df[f"m0_change_{lag}"]

    df["m0_diff"] = df.groupby("Date")["M0"].diff()

    # Stricter past-looking volatility:
    # shift(1) means only previous trades are used.
    df["m0_volatility_20"] = (
        df.groupby("Date")["m0_diff"]
        .transform(lambda x: x.shift(1).rolling(window=20, min_periods=5).std())
    )

    df["m0_volatility_50"] = (
        df.groupby("Date")["m0_diff"]
        .transform(lambda x: x.shift(1).rolling(window=50, min_periods=10).std())
    )

    df["signed_flow_unit"] = df["Side"] * df["log_volume"]

    # Stricter past-looking signed flow:
    # current trade is not included in its own flow history.
    df["signed_flow_20"] = (
        df.groupby("Date")["signed_flow_unit"]
        .transform(lambda x: x.shift(1).rolling(window=20, min_periods=1).sum())
    )

    df["signed_flow_50"] = (
        df.groupby("Date")["signed_flow_unit"]
        .transform(lambda x: x.shift(1).rolling(window=50, min_periods=1).sum())
    )

    df["side_signed_flow_20"] = df["Side"] * df["signed_flow_20"]
    df["side_signed_flow_50"] = df["Side"] * df["signed_flow_50"]

    # Stricter past-looking spread history.
    df["spread_roll_mean_20"] = (
        df.groupby("Date")["Spread"]
        .transform(lambda x: x.shift(1).rolling(window=20, min_periods=5).mean())
    )

    df["spread_roll_mean_50"] = (
        df.groupby("Date")["Spread"]
        .transform(lambda x: x.shift(1).rolling(window=50, min_periods=10).mean())
    )

    df["spread_ratio_20"] = df["Spread"] / df["spread_roll_mean_20"]
    df["spread_ratio_50"] = df["Spread"] / df["spread_roll_mean_50"]

    df = df.drop(columns=["m0_diff", "signed_flow_unit"])

    return df


def split_by_date(df):
    unique_dates = sorted(df["Date"].unique())
    n_dates = len(unique_dates)

    train_end = int(0.60 * n_dates)
    val_end = int(0.80 * n_dates)

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    train_df = df[df["Date"].isin(train_dates)].copy()
    val_df = df[df["Date"].isin(val_dates)].copy()
    test_df = df[df["Date"].isin(test_dates)].copy()

    print("Date split:")
    print(
        "Train:",
        train_dates[0],
        "to",
        train_dates[-1],
        "|",
        len(train_dates),
        "dates",
    )
    print(
        "Validation:",
        val_dates[0],
        "to",
        val_dates[-1],
        "|",
        len(val_dates),
        "dates",
    )
    print(
        "Test:",
        test_dates[0],
        "to",
        test_dates[-1],
        "|",
        len(test_dates),
        "dates",
    )

    return train_df, val_df, test_df


def expand_by_horizon(df):
    expanded_parts = []

    for tau in HORIZONS:
        temp = df.copy()
        future_col = f"M{tau}"

        temp["tau"] = tau
        temp["tau_scaled"] = tau / 30.0

        # LP perspective:
        # PnL(tau) = Side * Volume * (M_tau - Trade Price)
        # Adverse = 1 if LP PnL < 0.
        temp["adverse"] = (
            temp["Side"] * (temp[future_col] - temp["Trade Price"]) < 0
        ).astype(int)

        temp["volume_x_tau"] = temp["log_volume"] * temp["tau_scaled"]
        temp["spread_x_tau"] = temp["relative_spread"] * temp["tau_scaled"]

        temp["side_momentum_20_x_tau"] = (
            temp["side_m0_change_20"] * temp["tau_scaled"]
        )

        temp["side_flow_50_x_tau"] = (
            temp["side_signed_flow_50"] * temp["tau_scaled"]
        )

        expanded_parts.append(temp)

    expanded = pd.concat(expanded_parts, ignore_index=True)
    return expanded


def build_train_only_rate_tables(train_df):
    global_rate = train_df["adverse"].mean()

    rate_tables = {
        "global_rate": global_rate,
        "client_rate": (
            train_df.groupby("Name")["adverse"]
            .mean()
            .reset_index()
            .rename(columns={"adverse": "client_adverse_rate"})
        ),
        "client_tau_rate": (
            train_df.groupby(["Name", "tau"])["adverse"]
            .mean()
            .reset_index()
            .rename(columns={"adverse": "client_tau_adverse_rate"})
        ),
        "client_side_rate": (
            train_df.groupby(["Name", "Side"])["adverse"]
            .mean()
            .reset_index()
            .rename(columns={"adverse": "client_side_adverse_rate"})
        ),
        "client_side_tau_rate": (
            train_df.groupby(["Name", "Side", "tau"])["adverse"]
            .mean()
            .reset_index()
            .rename(columns={"adverse": "client_side_tau_adverse_rate"})
        ),
        "side_tau_rate": (
            train_df.groupby(["Side", "tau"])["adverse"]
            .mean()
            .reset_index()
            .rename(columns={"adverse": "side_tau_adverse_rate"})
        ),
    }

    return rate_tables


def add_train_only_client_features(df, rate_tables):
    df = df.copy()
    global_rate = rate_tables["global_rate"]

    df = df.merge(rate_tables["client_rate"], on="Name", how="left")
    df = df.merge(rate_tables["client_tau_rate"], on=["Name", "tau"], how="left")
    df = df.merge(rate_tables["client_side_rate"], on=["Name", "Side"], how="left")

    df = df.merge(
        rate_tables["client_side_tau_rate"],
        on=["Name", "Side", "tau"],
        how="left",
    )

    df = df.merge(rate_tables["side_tau_rate"], on=["Side", "tau"], how="left")

    rate_cols = [
        "client_adverse_rate",
        "client_tau_adverse_rate",
        "client_side_adverse_rate",
        "client_side_tau_adverse_rate",
        "side_tau_adverse_rate",
    ]

    for col in rate_cols:
        df[col] = df[col].fillna(global_rate)

    return df


def get_feature_columns():
    categorical_features = ["Name"]

    numeric_features = [
        "Side",
        "Volume",
        "Trade Price",
        "M0",
        "Spread",
        "hour",
        "minute",
        "second",
        "day_of_week",
        "seconds_from_midnight",
        "tau",
        "tau_scaled",
        "price_vs_mid",
        "abs_price_vs_mid",
        "relative_spread",
        "price_vs_mid_in_spreads",
        "abs_price_vs_mid_in_spreads",
        "side_x_price_vs_mid",
        "side_x_price_vs_mid_in_spreads",
        "signed_volume",
        "log_volume",
        "side_x_volume",
        "m0_change_10",
        "m0_change_20",
        "m0_change_50",
        "side_m0_change_10",
        "side_m0_change_20",
        "side_m0_change_50",
        "m0_volatility_20",
        "m0_volatility_50",
        "signed_flow_20",
        "signed_flow_50",
        "side_signed_flow_20",
        "side_signed_flow_50",
        "spread_roll_mean_20",
        "spread_roll_mean_50",
        "spread_ratio_20",
        "spread_ratio_50",
        "volume_x_tau",
        "spread_x_tau",
        "side_momentum_20_x_tau",
        "side_flow_50_x_tau",
        "client_adverse_rate",
        "client_tau_adverse_rate",
        "client_side_adverse_rate",
        "client_side_tau_adverse_rate",
        "side_tau_adverse_rate",
    ]

    return categorical_features, numeric_features


def build_histgb_model(categorical_features, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", SimpleImputer(strategy="median"), numeric_features),
        ]
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.025,
        max_iter=450,
        max_leaf_nodes=31,
        min_samples_leaf=120,
        l2_regularization=0.10,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_extratrees_model(categorical_features, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", SimpleImputer(strategy="median"), numeric_features),
        ]
    )

    model = ExtraTreesClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=150,
        max_features="sqrt",
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_probabilities(y_true, prob, threshold=0.5):
    prob = np.clip(prob, 0.001, 0.999)
    pred = (prob >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "log_loss": log_loss(y_true, prob),
    }


def train_horizon_models(train_df, val_df, categorical_features, numeric_features):
    feature_columns = categorical_features + numeric_features

    models_by_tau = {}
    model_selection_rows = []

    for tau in HORIZONS:
        print(f"\nTraining horizon-specific models for tau={tau}...")

        train_tau = train_df[train_df["tau"] == tau].copy()
        val_tau = val_df[val_df["tau"] == tau].copy()

        X_train = train_tau[feature_columns]
        y_train = train_tau["adverse"]

        X_val = val_tau[feature_columns]
        y_val = val_tau["adverse"]

        candidates = {
            "histgb": build_histgb_model(categorical_features, numeric_features),
            "extratrees": build_extratrees_model(categorical_features, numeric_features),
        }

        best_name = None
        best_model = None
        best_val_logloss = float("inf")

        for model_name, model in candidates.items():
            print(f"  Fitting {model_name}...")

            model.fit(X_train, y_train)

            val_prob = model.predict_proba(X_val)[:, 1]
            val_metrics = evaluate_probabilities(y_val, val_prob)

            model_selection_rows.append(
                {
                    "tau": tau,
                    "model": model_name,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_log_loss": val_metrics["log_loss"],
                }
            )

            print(
                f"  {model_name}: "
                f"val_acc={val_metrics['accuracy']:.6f}, "
                f"val_log_loss={val_metrics['log_loss']:.6f}"
            )

            if val_metrics["log_loss"] < best_val_logloss:
                best_val_logloss = val_metrics["log_loss"]
                best_name = model_name
                best_model = model

        models_by_tau[tau] = best_model

        print(
            f"Best model for tau={tau}: {best_name} "
            f"| val_log_loss={best_val_logloss:.6f}"
        )

    model_selection_df = pd.DataFrame(model_selection_rows)
    return models_by_tau, model_selection_df


def evaluate_horizon_models(
    models_by_tau,
    df,
    split_name,
    categorical_features,
    numeric_features,
):
    feature_columns = categorical_features + numeric_features

    horizon_rows = []

    for tau in HORIZONS:
        temp = df[df["tau"] == tau].copy()

        X = temp[feature_columns]
        y_true = temp["adverse"]

        model = models_by_tau[tau]
        prob = model.predict_proba(X)[:, 1]

        metrics = evaluate_probabilities(y_true, prob)

        horizon_rows.append(
            {
                "split": split_name,
                "tau": tau,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "log_loss": metrics["log_loss"],
            }
        )

    horizon_df = pd.DataFrame(horizon_rows)

    avg_result = {
        "split": split_name,
        "accuracy": horizon_df["accuracy"].mean(),
        "precision": horizon_df["precision"].mean(),
        "recall": horizon_df["recall"].mean(),
        "log_loss": horizon_df["log_loss"].mean(),
    }

    return avg_result, horizon_df


def train_pipeline(data_path="trade_data.csv"):
    global MODELS_BY_TAU, ARTIFACTS

    trade_data = load_data(data_path)
    print("Loaded trade_data:", trade_data.shape)

    trade_data = add_trade_features(trade_data)

    train_raw, val_raw, test_raw = split_by_date(trade_data)

    print("\nRaw split sizes:")
    print("Train:", train_raw.shape)
    print("Validation:", val_raw.shape)
    print("Test:", test_raw.shape)

    train_df = expand_by_horizon(train_raw)
    val_df = expand_by_horizon(val_raw)
    test_df = expand_by_horizon(test_raw)

    rate_tables = build_train_only_rate_tables(train_df)

    train_df = add_train_only_client_features(train_df, rate_tables)
    val_df = add_train_only_client_features(val_df, rate_tables)
    test_df = add_train_only_client_features(test_df, rate_tables)

    print("\nExpanded split sizes:")
    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    categorical_features, numeric_features = get_feature_columns()

    models_by_tau, model_selection_df = train_horizon_models(
        train_df,
        val_df,
        categorical_features,
        numeric_features,
    )

    MODELS_BY_TAU = models_by_tau

    ARTIFACTS = {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "rate_tables": rate_tables,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "model_selection_df": model_selection_df,
    }

    return ARTIFACTS


def compute_metrics(*args, **kwargs) -> pd.DataFrame:
    if not ARTIFACTS:
        data_path = kwargs.get("data_path", "trade_data.csv")
        train_pipeline(data_path=data_path)

    categorical_features = ARTIFACTS["categorical_features"]
    numeric_features = ARTIFACTS["numeric_features"]

    results = []
    horizon_outputs = []

    split_map = {
        "train": ARTIFACTS["train_df"],
        "validation": ARTIFACTS["val_df"],
        "test": ARTIFACTS["test_df"],
    }

    for split_name, split_df in split_map.items():
        avg_result, horizon_df = evaluate_horizon_models(
            MODELS_BY_TAU,
            split_df,
            split_name,
            categorical_features,
            numeric_features,
        )

        results.append(avg_result)
        horizon_outputs.append(horizon_df)

    results_df = pd.DataFrame(results)
    horizon_results_df = pd.concat(horizon_outputs, ignore_index=True)

    results_df.to_csv("task3_results.csv", index=False)
    horizon_results_df.to_csv("task3_horizon_results.csv", index=False)

    ARTIFACTS["horizon_results_df"] = horizon_results_df

    return results_df


def _prepare_single_prediction_row(row_data):
    if not ARTIFACTS:
        train_pipeline()

    if isinstance(row_data, pd.Series):
        row = row_data.to_dict()
    elif isinstance(row_data, dict):
        row = dict(row_data)
    else:
        raise ValueError(
            "predict_adversity expects a dict, pandas Series, or keyword arguments."
        )

    tau = int(row.get("tau", row.get("horizon", 30)))

    base = pd.DataFrame([row])

    if "Date" not in base.columns:
        base["Date"] = ARTIFACTS["train_df"]["Date"].iloc[-1]

    if "time" not in base.columns:
        base["time"] = "00:00:00"

    required_defaults = {
        "Name": ARTIFACTS["train_df"]["Name"].mode().iloc[0],
        "Side": 1,
        "Volume": ARTIFACTS["train_df"]["Volume"].median(),
        "Trade Price": ARTIFACTS["train_df"]["Trade Price"].median(),
        "M0": ARTIFACTS["train_df"]["M0"].median(),
        "Spread": ARTIFACTS["train_df"]["Spread"].median(),
    }

    for col, default_value in required_defaults.items():
        if col not in base.columns:
            base[col] = default_value

    base = add_trade_features(base)

    base["tau"] = tau
    base["tau_scaled"] = tau / 30.0

    base["volume_x_tau"] = base["log_volume"] * base["tau_scaled"]
    base["spread_x_tau"] = base["relative_spread"] * base["tau_scaled"]
    base["side_momentum_20_x_tau"] = base["side_m0_change_20"] * base["tau_scaled"]
    base["side_flow_50_x_tau"] = base["side_signed_flow_50"] * base["tau_scaled"]

    base = add_train_only_client_features(base, ARTIFACTS["rate_tables"])

    return base, tau


def predict_adversity(*args, **kwargs) -> float:
    if not ARTIFACTS:
        train_pipeline()

    if args:
        row_data = args[0]
    else:
        row_data = kwargs

    row_df, tau = _prepare_single_prediction_row(row_data)

    if tau not in MODELS_BY_TAU:
        raise ValueError(f"tau must be one of {HORIZONS}")

    categorical_features = ARTIFACTS["categorical_features"]
    numeric_features = ARTIFACTS["numeric_features"]
    feature_columns = categorical_features + numeric_features

    model = MODELS_BY_TAU[tau]
    probability = model.predict_proba(row_df[feature_columns])[:, 1][0]

    return float(np.clip(probability, 0.001, 0.999))


# ============================================================
# TASK 4 EXTERNALIZATION THRESHOLD LOGIC
# ============================================================

def _ensure_task4_ready(data_path="trade_data.csv"):
    """
    Ensures Task 3 model pipeline is trained before Task 4 calculations.
    """
    if not ARTIFACTS or not MODELS_BY_TAU:
        train_pipeline(data_path=data_path)


def _feature_columns():
    categorical_features = ARTIFACTS["categorical_features"]
    numeric_features = ARTIFACTS["numeric_features"]
    return categorical_features + numeric_features


def add_pnl_column(df, tau):
    """
    Adds LP-perspective PnL for one horizon.

    PnL_tau = Side * Volume * (M_tau - Trade Price)
    """
    df = df.copy()
    future_col = f"M{tau}"

    if future_col not in df.columns:
        raise ValueError(f"Missing required column: {future_col}")

    df["pnl_tau"] = (
        df["Side"] * df["Volume"] * (df[future_col] - df["Trade Price"])
    )

    return df


def get_probabilities_for_split(split_df, tau):
    """
    Predicts Task 3 adversity probabilities for a given split and horizon.
    """
    temp = split_df[split_df["tau"] == tau].copy()

    if temp.empty:
        raise ValueError(f"No rows found for tau={tau}")

    feature_cols = _feature_columns()
    model = MODELS_BY_TAU[tau]

    probabilities = model.predict_proba(temp[feature_cols])[:, 1]
    probabilities = np.clip(probabilities, 0.001, 0.999)

    temp = add_pnl_column(temp, tau)
    temp["adversity_probability"] = probabilities

    return temp


def strategy_pnl_from_threshold(df, theta):
    """
    Computes strategy PnL under externalization rule.

    If adversity_probability > theta:
        externalize -> PnL = 0

    Else:
        internalize -> PnL = actual LP PnL
    """
    externalized = df["adversity_probability"] > theta

    strategy_pnl = np.where(
        externalized,
        0.0,
        df["pnl_tau"].values
    )

    return float(strategy_pnl.sum())


def find_best_threshold(df, theta_values=None):
    """
    Finds theta that maximizes validation PnL.
    """
    if theta_values is None:
        theta_values = np.linspace(0.0, 1.0, 101)

    best_theta = None
    best_pnl = -np.inf

    curve_rows = []

    for theta in theta_values:
        pnl = strategy_pnl_from_threshold(df, theta)

        curve_rows.append(
            {
                "theta": theta,
                "validation_pnl": pnl,
            }
        )

        if pnl > best_pnl:
            best_pnl = pnl
            best_theta = theta

    curve_df = pd.DataFrame(curve_rows)

    return float(best_theta), float(best_pnl), curve_df


def optimal_threshold(tau, client=None, data_path="trade_data.csv") -> dict:
    """
    Finds optimal externalization threshold.

    Parameters:
    tau:
        Horizon, one of [5, 10, 15, 20, 25, 30]

    client:
        If provided, finds client-specific threshold.
        If None, finds global threshold across all clients.

    Returns:
        {
            "theta": float,
            "validation_pnl": float,
            "test_pnl": float
        }
    """
    _ensure_task4_ready(data_path=data_path)

    if tau not in HORIZONS:
        raise ValueError(f"tau must be one of {HORIZONS}")

    val_df = get_probabilities_for_split(ARTIFACTS["val_df"], tau)
    test_df = get_probabilities_for_split(ARTIFACTS["test_df"], tau)

    if client is not None:
        val_df = val_df[val_df["Name"] == client].copy()
        test_df = test_df[test_df["Name"] == client].copy()

        if val_df.empty:
            raise ValueError(f"No validation rows found for client={client}, tau={tau}")

        if test_df.empty:
            raise ValueError(f"No test rows found for client={client}, tau={tau}")

    theta_star, validation_pnl, _ = find_best_threshold(val_df)
    test_pnl = strategy_pnl_from_threshold(test_df, theta_star)

    return {
        "theta": theta_star,
        "validation_pnl": validation_pnl,
        "test_pnl": float(test_pnl),
    }


def plot_pnl_vs_theta(tau=None, data_path="trade_data.csv") -> None:
    """
    Plots validation PnL versus theta.

    If tau is None, plots one curve per horizon.
    Saves figure to pnl_vs_theta.png.
    """
    _ensure_task4_ready(data_path=data_path)

    theta_values = np.linspace(0.0, 1.0, 101)

    plt.figure(figsize=(10, 6))

    horizons_to_plot = HORIZONS if tau is None else [tau]

    all_curve_rows = []

    for current_tau in horizons_to_plot:
        val_df = get_probabilities_for_split(ARTIFACTS["val_df"], current_tau)

        curve_rows = []

        for theta in theta_values:
            pnl = strategy_pnl_from_threshold(val_df, theta)

            curve_rows.append(
                {
                    "tau": current_tau,
                    "theta": theta,
                    "validation_pnl": pnl,
                }
            )

        curve_df = pd.DataFrame(curve_rows)
        all_curve_rows.append(curve_df)

        plt.plot(
            curve_df["theta"],
            curve_df["validation_pnl"],
            label=f"tau={current_tau}",
        )

    final_curve_df = pd.concat(all_curve_rows, ignore_index=True)
    final_curve_df.to_csv("task4_threshold_curve.csv", index=False)

    plt.xlabel("Externalization threshold theta")
    plt.ylabel("Validation PnL")
    plt.title("Validation PnL vs Externalization Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pnl_vs_theta.png", dpi=300)
    plt.close()

    print("Saved: pnl_vs_theta.png")
    print("Saved: task4_threshold_curve.csv")


def generate_task4_results(data_path="trade_data.csv"):
    """
    Generates client-specific optimal threshold for every client and horizon.

    Saves task4_results.csv with required columns:
    client, tau, theta_star, final_pnl
    """
    _ensure_task4_ready(data_path=data_path)

    clients = sorted(ARTIFACTS["val_df"]["Name"].unique())

    result_rows = []

    for client in clients:
        for tau in HORIZONS:
            result = optimal_threshold(
                tau=tau,
                client=client,
                data_path=data_path,
            )

            result_rows.append(
                {
                    "client": client,
                    "tau": tau,
                    "theta_star": result["theta"],
                    "final_pnl": result["test_pnl"],
                }
            )

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv("task4_results.csv", index=False)

    return results_df


def summarize_task4_results(results_df):
    print("\nTask 4 Results")
    print(results_df)

    print("\nTotal final test PnL across all client-horizon rows:")
    print(results_df["final_pnl"].sum())

    print("\nAverage theta by horizon:")
    print(results_df.groupby("tau")["theta_star"].mean())

    print("\nAverage theta by client:")
    print(results_df.groupby("client")["theta_star"].mean())


# ============================================================
# MAIN
# ============================================================

def main():
    data_path = "trade_data.csv"

    print("Running Task 3 model pipeline for Task 4...")
    train_pipeline(data_path=data_path)

    print("\nComputing Task 3 metrics...")
    task3_metrics = compute_metrics(data_path=data_path)
    print(task3_metrics)

    ARTIFACTS["model_selection_df"].to_csv("task3_model_selection.csv", index=False)

    print("\nGenerating Task 4 client-specific thresholds...")
    task4_results = generate_task4_results(data_path=data_path)

    summarize_task4_results(task4_results)

    print("\nGenerating PnL vs theta plot...")
    plot_pnl_vs_theta(tau=None, data_path=data_path)

    print("\nSaved files:")
    print("Saved: task3_results.csv")
    print("Saved: task3_horizon_results.csv")
    print("Saved: task3_model_selection.csv")
    print("Saved: task4_results.csv")
    print("Saved: task4_threshold_curve.csv")
    print("Saved: pnl_vs_theta.png")


if __name__ == "__main__":
    main()