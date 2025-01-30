import os
import sys
import warnings
from typing import List, Dict, Any, Union

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import t, norm

# Unused modules are commented out
# from aiml.model_param import BaseModel
# from common.trading_logger import TradingLogger

warnings.filterwarnings("ignore", category=FutureWarning, module="arch")

# Add the parent directory to the Python path to import modules from mongodb
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import MARKET_DATA_TECH


class GarchPriceDistributionModel:
    """
    Sample class to predict BTC price return volatility using GARCH(1,1)
    and estimate price distribution (quantiles).
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        mean: str = "constant",
        vol: str = "GARCH",
        quantiles: List[float] = None
    ) -> None:
        """
        Initializes the GarchPriceDistributionModel.

        Parameters
        ----------
        p : int, default 1
            The order of the GARCH term.
        q : int, default 1
            The order of the ARCH term.
        dist : str, default "normal"
            The distribution of the residuals ('normal', 't', etc.).
        mean : str, default "constant"
            The model for the mean ('constant', 'AR', 'HAR', etc.).
        vol : str, default "GARCH"
            The volatility model ('GARCH', 'EGARCH', 'ARCH', etc.).
        quantiles : List[float], optional
            List of quantiles to predict. Defaults to [0.10, 0.5, 0.90].
        """
        if quantiles is None:
            quantiles = [0.10, 0.5, 0.90]

        self.p = p
        self.q = q
        self.dist = dist
        self.mean = mean
        self.vol = vol
        self.model_fit = None  # Stores the fitted GARCH model
        self.last_price: Union[float, None] = None  # Stores the last observed price
        self.quantiles = quantiles
        self.df_model: Union[pd.DataFrame, None] = None  # Stores the data used for modeling


    def prepare_data(
        self, df: pd.DataFrame, price_col: str = "close"
    ) -> pd.Series:
        """
        Prepares the data by calculating log returns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with price data.
        price_col : str, default "close"
            Name of the price column.

        Returns
        -------
        returns : pd.Series
            Calculated log returns.
        """
        if "date" in df.columns:
            df = df.set_index("date")

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df = df.copy()

        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
        df.dropna(subset=["log_return"], inplace=True)

        self.last_price = df[price_col].iloc[-1]
        self.df_model = df.copy()
        return df["log_return"]

    def fit_model(self, returns: pd.Series) -> None:
        """
        Fits the GARCH model to the given returns series.

        Parameters
        ----------
        returns : pd.Series
            Log return series.
        """
        am = arch_model(
            returns,
            mean=self.mean,
            vol=self.vol,
            p=self.p,
            q=self.q,
            dist=self.dist,
            rescale=False
        )
        self.model_fit = am.fit(disp="off")

    def forecast_distribution(
        self, horizon: int = 1, quantiles: List[float] = None
    ) -> Dict[str, Any]:
        """
        Forecasts the return and price distribution based on quantiles.

        Parameters
        ----------
        horizon : int, default 1
            Forecast horizon (e.g., 1 for 1 day ahead).
        quantiles : List[float], optional
            List of quantiles to predict. If None, uses the instance variable.

        Returns
        -------
        dict
            A dictionary containing the forecasted return quantiles,
            price quantiles, and the forecast horizon.
        """
        if quantiles is None:
            quantiles = self.quantiles

        if self.model_fit is None:
            raise ValueError("Model is not fitted yet. Please call fit_model() first.")

        forecast_res = self.model_fit.forecast(horizon=horizon)
        mean_forecast = forecast_res.mean[f"h.{horizon}"].iloc[-1]
        var_forecast = forecast_res.variance[f"h.{horizon}"].iloc[-1]
        std_forecast = np.sqrt(var_forecast)

        # Extract degrees of freedom for t-distribution
        if self.dist == "t":
            nu = self.model_fit.params.get("nu", 10.0)
        else:
            nu = None

        ret_quantiles = {}
        price_quantiles = {}
        current_price = self.last_price

        for q in quantiles:
            if self.dist == "t":
                z_q = t.ppf(q, df=nu)
            else:
                z_q = norm.ppf(q)

            ret_q = mean_forecast + std_forecast * z_q
            ret_quantiles[q] = ret_q

            price_q = current_price * np.exp(ret_q)
            price_quantiles[q] = price_q

        return {
            "forecast_returns_quantiles": ret_quantiles,
            "forecast_price_quantiles": price_quantiles,
            "horizon": horizon
        }

    def forecast_volatility_for_specific_date(
        self,
        df: pd.DataFrame,
        horizon: int = 1
    ) -> float:
        """
        Estimates and returns the volatility for a specified date using a GARCH(1,1) model.
        Automatically calculates the horizon as the number of days between the last observation
        in df and the target_date.

        Parameters
        ----------
        df : pd.DataFrame
            Historical price data (DataFrame with date index)
        target_date : str
            Target date for prediction (e.g., "2025-01-01")
        price_col : str, default "close"
            Name of the price column

        Returns
        -------
        float
            Predicted volatility (standard deviation) for the specified date
        """
        # 1) Sort DataFrame by date if needed
        df = df.sort_index()


        if horizon <= 0:
            raise ValueError(f"Horizon must be greater than 0")

        # 5) Create returns for model input
        returns = self.prepare_data(df)

        # 6) Train GARCH model
        self.fit_model(returns)

        # 7) Forecast variance/volatility for horizon days ahead
        forecast_res = self.model_fit.forecast(horizon=horizon)

        # Get variance for h days ahead → volatility = sqrt(variance)
        var_forecast = forecast_res.variance[f"h.{horizon}"].iloc[-1]
        std_forecast = float(var_forecast**0.5)

        return std_forecast


def judge_current_price(current_price: float, lower_bound: float, upper_bound: float) -> str:
    """
    Judges the current price as potentially undervalued, fair, or overvalued
    based on the predicted lower and upper bounds. (Immediate Judgement)

    Parameters
    ----------
    current_price : float
        Current price.
    lower_bound : float
        Lower bound price (e.g., 10% quantile).
    upper_bound : float
        Upper bound price (e.g., 90% quantile).

    Returns
    -------
    str
        "POTENTIALLY_UNDERVALUED", "POTENTIALLY_FAIR", or "POTENTIALLY_OVERVALUED".
    """
    if current_price < lower_bound:
        return "POTENTIALLY_UNDERVALUED"
    elif current_price > upper_bound:
        return "POTENTIALLY_OVERVALUED"
    return "POTENTIALLY_FAIR"

def judge_next_day_price(
    actual_next_price: float,
    lower_bound: float,
    upper_bound: float
) -> str:
    """
    Judges the next day's actual price as undervalued, fair, or overvalued
    based on the predicted lower and upper bounds. (Ex-post Validation)

    Parameters
    ----------
    actual_next_price : float
        The actual price on the next day.
    lower_bound : float
        Lower bound price.
    upper_bound : float
        Upper bound price.

    Returns
    -------
    str
        "UNDERVALUED", "FAIR", or "OVERVALUED".
    """
    if actual_next_price < lower_bound:
        return "UNDERVALUED"
    elif actual_next_price > upper_bound:
        return "OVERVALUED"
    return "FAIR"


def evaluate_predictions(df_result: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates the accuracy of immediate price judgements.

    Parameters
    ----------
    df_result : pd.DataFrame
        DataFrame containing the judgement results.

    Returns
    -------
    dict
        A dictionary containing the overall accuracy, confusion matrix, and
        transition matrix.
    """
    # Mapping for judgement results (removing "POTENTIALLY" prefix for comparison)
    mapping = {
        "POTENTIALLY_UNDERVALUED": "UNDERVALUED",
        "POTENTIALLY_FAIR": "FAIR",
        "POTENTIALLY_OVERVALUED": "OVERVALUED"
    }

    # Compare predicted and actual results
    total = len(df_result)
    correct = sum(
        df_result.apply(
            lambda row: mapping[row["current_judgement"]]
            == row["next_day_judgement"],
            axis=1
        )
    )
    accuracy = correct / total if total > 0 else 0.0

    # Create confusion matrix
    print("\n=== Immediate Judgement Accuracy Evaluation ===")
    print(f"Overall Accuracy: {accuracy:.1%}")

    print("\n=== Detailed Analysis ===")
    confusion = {
        "UNDERVALUED": {"correct": 0, "total": 0},
        "FAIR": {"correct": 0, "total": 0},
        "OVERVALUED": {"correct": 0, "total": 0}
    }

    for _, row in df_result.iterrows():
        predicted = mapping[row["current_judgement"]]
        actual = row["next_day_judgement"]

        confusion[predicted]["total"] += 1
        if predicted == actual:
            confusion[predicted]["correct"] += 1

    print("\nAccuracy of Each Judgement:")
    for judgement, stats in confusion.items():
        if stats["total"] > 0:
            judgement_accuracy = stats["correct"] / stats["total"]
            print(f"{judgement}: {stats['correct']}/{stats['total']} = {judgement_accuracy:.1%}")

    # Judgement transition analysis
    print("\n=== Judgement Transition Analysis ===")
    transitions = pd.crosstab(
        df_result["current_judgement"].map(mapping),
        df_result["next_day_judgement"],
        normalize="index"
    )
    print("\nTransition Probabilities of Immediate Judgements:")
    print(transitions.round(3))

    return {
        "overall_accuracy": accuracy,
        "confusion": confusion,
        "transitions": transitions
    }




def garch_price_distribution_analysis(
    current_date: str,
    price_col: str = "close",
    quantiles: List[float] = None,
    symbol: str = "BTCUSDT",
    interval_min: int = 1440,
    horizon: int = 1
) -> Dict[str, Any]:
    """
    Analyzes price distribution using the GARCH model.

    Parameters
    ----------
    current_date : str
        Current date for analysis.
    price_col : str, default "close"
        Name of the price column.
    quantiles : List[float], optional
        List of quantiles to predict. Defaults to [0.10, 0.5, 0.90].
    symbol : str, default "BTCUSDT"
        Symbol name (e.g., "BTCUSDT").
    interval_min : int, default 1440
        Candlestick interval in minutes.

    Returns
    -------
    dict
        The return value of forecast_distribution().
    """
    if quantiles is None:
        quantiles = [0.10, 0.5, 0.90]

    data_loader = MongoDataLoader()
    df = data_loader.load_data_from_datetime_period(
        start_date="2020-01-01 00:00:00",
        end_date=current_date,
        coll_type=MARKET_DATA_TECH,
        symbol=symbol,
        interval=interval_min
    )

    # Sort by date and set date as index
    df = df.sort_values("date").reset_index(drop=True)
    df.set_index("date", inplace=True)

    model = GarchPriceDistributionModel(dist="t", quantiles=quantiles)

    # Extract the last 200 days of data, excluding the current date
    df_window = df.iloc[-200:-1]

    returns = model.prepare_data(df_window, price_col=price_col)
    model.fit_model(returns)
    result = model.forecast_distribution(horizon=horizon)

    return result

def garch_volatility_forecast(
    current_date: str,
    quantiles: List[float] = None,
    horizon: int = 1,
    symbol: str = "BTCUSDT",
    interval_min: int = 1440
) -> float:
    """
    Forecasts the volatility for a specified date using a GARCH(1,1) model.
    """
    data_loader = MongoDataLoader()
    df = data_loader.load_data_from_datetime_period(
        start_date="2020-01-01 00:00:00",
        end_date=current_date,
        coll_type=MARKET_DATA_TECH,
        symbol=symbol,
        interval=interval_min
    )
    model = GarchPriceDistributionModel(dist="t", quantiles=quantiles)
    returns = model.prepare_data(df)
    model.fit_model(returns)
    result = model.forecast_volatility_for_specific_date(df, horizon=horizon)
    return result

def get_current_price(current_date_str: str, interval_min: int) -> float:
    """
    Get the current price from the database
    """
    data_loader = MongoDataLoader()
    df = data_loader.load_data_from_datetime_period(
        start_date=current_date_str,
        end_date=current_date_str,
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=interval_min
    )
    return df["close"].iloc[-1]

def main() -> None:
    """
    Main function: Demonstrates one-day ahead price prediction and
    immediate/ex-post judgement using GARCH on BTCUSDT daily data.
    """
    m_interval = 1440

    # 1. Load data
    data_loader = MongoDataLoader()
    df = data_loader.load_data(
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=m_interval
    )

    df = df.sort_values("date").reset_index(drop=True)
    df.set_index("date", inplace=True)

    # 2. Specify the period for judgement (example)
    start_date_judge = "2024-04-01"
    end_date_judge = "2025-01-01"

    # Create DataFrame for judgement
    df_for_judge = df.loc[start_date_judge: end_date_judge]

    # Parameter settings
    lookback = 200  # Number of past days to use for training
    horizon = 3     # Predict 1 day ahead
    quantiles = [0.10, 0.5, 0.90]

    # 3. Perform judgement for each day within the specified period
    judge_indices = df_for_judge.index.tolist()
    result_judgement = []

    for current_date in judge_indices:
        # Get the row number of current_date within the entire df
        i = df.index.get_loc(current_date)

        # Skip if not enough lookback data is available
        if i - lookback < 0:
            continue

        # Skip if not enough future data is available for the horizon
        if i + horizon >= len(df):
            break

        # Extract data for the past lookback days
        df_window = df.iloc[i - lookback: i]

        # Create and train the model
        model = GarchPriceDistributionModel(dist="t", quantiles=quantiles)
        returns = model.prepare_data(df_window, price_col="close")
        model.fit_model(returns)

        # Get predicted quantiles
        result_forecast = model.forecast_distribution(horizon=horizon)
        forecast_price_quantiles = result_forecast["forecast_price_quantiles"]

        lower_bound = forecast_price_quantiles[model.quantiles[0]]
        median_val = forecast_price_quantiles[model.quantiles[1]]
        upper_bound = forecast_price_quantiles[model.quantiles[2]]

        # Perform immediate judgement
        current_judgement = judge_current_price(
            current_price=df["close"].iloc[i],
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )

        # Perform ex-post judgement
        next_day_judgement = judge_next_day_price(
            actual_next_price=df["close"].iloc[i + horizon],
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )

        next_date = df.index[i + horizon]
        result_judgement.append({
            "current_date": current_date,
            "next_date": next_date,
            f"lower_{int(model.quantiles[0]*100)}pct": lower_bound,
            f"median_{int(model.quantiles[1]*100)}pct": median_val,
            f"upper_{int(model.quantiles[2]*100)}pct": upper_bound,
            "current_price": df["close"].iloc[i],
            "next_day_price": df["close"].iloc[i + horizon],
            "current_judgement": current_judgement,
            "next_day_judgement": next_day_judgement
        })

    # Convert to DataFrame
    df_result = pd.DataFrame(result_judgement)

    # Summarize immediate judgements
    print("\n=== Immediate Judgement Results ===")
    current_counts = df_result["current_judgement"].value_counts()
    total = len(df_result)
    print(f"Potentially Undervalued (POTENTIALLY_UNDERVALUED): {current_counts.get('POTENTIALLY_UNDERVALUED', 0)} "
          f"({(current_counts.get('POTENTIALLY_UNDERVALUED', 0) / total * 100):.1f}%)")
    print(f"Potentially Fair (POTENTIALLY_FAIR): {current_counts.get('POTENTIALLY_FAIR', 0)} "
          f"({(current_counts.get('POTENTIALLY_FAIR', 0) / total * 100):.1f}%)")
    print(f"Potentially Overvalued (POTENTIALLY_OVERVALUED): {current_counts.get('POTENTIALLY_OVERVALUED', 0)} "
          f"({(current_counts.get('POTENTIALLY_OVERVALUED', 0) / total * 100):.1f}%)")


    # Summarize ex-post validation results
    print("\n=== Ex-post Validation Results ===")
    next_counts = df_result["next_day_judgement"].value_counts()
    print(f"Undervalued (UNDERVALUED): {next_counts.get('UNDERVALUED', 0)} "
          f"({(next_counts.get('UNDERVALUED', 0) / total * 100):.1f}%)")
    print(f"Fair (FAIR): {next_counts.get('FAIR', 0)} "
          f"({(next_counts.get('FAIR', 0) / total * 100):.1f}%)")
    print(f"Overvalued (OVERVALUED): {next_counts.get('OVERVALUED', 0)} "
          f"({(next_counts.get('OVERVALUED', 0) / total * 100):.1f}%)")

    # Evaluate prediction accuracy
    _ = evaluate_predictions(df_result)

    # Example of calling the analysis function
    current_date_str = "2025-01-15 00:00:00"
    result = garch_price_distribution_analysis(current_date_str, horizon=horizon, interval_min=m_interval)
    print(result)

    volatility = garch_volatility_forecast(current_date_str, horizon=horizon, interval_min=m_interval)
    print(f"Volatility forecast for {current_date_str}: {volatility:.4f}")

    # Example: CSV output
    # df_result.to_csv("garch_judgement.csv", index=False)


if __name__ == "__main__":
    main()