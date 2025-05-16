"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """

        # Implement a rolling Mean-Variance Optimization strategy
        # Iterate through the index of the price DataFrame starting after the lookback period
        # Use iloc for integer-based indexing for the loop to avoid issues with missing dates
        for i in range(self.lookback, len(self.price.index)):
            current_date = self.price.index[i]

            # Define the lookback window using iloc and then get the dates
            lookback_start_iloc = max(0, i - self.lookback)
            lookback_end_iloc = i - 1 # Use data up to the day before the current date

            # Ensure lookback window is valid
            if lookback_end_iloc < lookback_start_iloc:
                 continue # Skip if lookback window is empty or invalid

            lookback_start_date = self.price.index[lookback_start_iloc]
            lookback_end_date = self.price.index[lookback_end_iloc]

            # Get prices for the lookback window using date slicing
            # Ensure there are enough data points in the price series for the lookback window
            # .loc handles potential missing dates in the index correctly
            # Use the correctly scoped assets_for_optimization here
            lookback_prices = self.price.loc[lookback_start_date:lookback_end_date, assets_for_optimization]

            # Calculate daily returns within the lookback window
            # Drop the first row of NaN from pct_change
            lookback_returns = lookback_prices.pct_change().dropna()

            # Need at least 2 data points (i.e., at least 3 price points) to calculate covariance
            if len(lookback_returns) < 2:
                # Not enough data in lookback window for covariance calculation
                # Weights for current_date remain NaN initially
                continue

            # Estimate expected returns (annualized)
            # Use mean of historical daily returns, then annualize
            expected_returns = lookback_returns.mean() * 252

            # Calculate covariance matrix (annualized assumes daily returns)
            # Check if lookback_returns has only one row - cov() would be all zeros
            if len(lookback_returns) > 1:
                 cov_matrix = lookback_returns.cov() * 252 # Annualized covariance
            else:
                 # Handle case with only one data point in lookback_returns
                 continue

            # Handle potential NaNs or Infs in covariance matrix
            if cov_matrix.isnull().values.any() or np.isinf(cov_matrix).values.any():
                print(f"Warning: Covariance matrix contains NaN or Inf on {current_date}. Skipping optimization.")
                continue

            # Ensure covariance matrix is positive semi-definite (optional but robust)
            # np.linalg.eigvalsh can check eigenvalues; all should be non-negative for PSD
            try:
                eigenvalues = np.linalg.eigvalsh(cov_matrix)
                # Check if any eigenvalue is negative (allowing a small tolerance for numerical stability)
                if np.any(eigenvalues < -1e-6):
                     print(f"Warning: Covariance matrix is not positive semi-definite on {current_date}. Skipping optimization.")
                     continue
            except np.linalg.LinAlgError:
                 # Handle cases where eigenvalue computation itself fails
                 print(f"Warning: Could not compute eigenvalues for covariance matrix on {current_date}. Skipping optimization.")
                 continue

            try:
                # Create a new Gurobi model for optimization
                model = gp.Model()
                model.setParam('OutputFlag', 0) # Suppress Gurobi output
                model.setParam('TimeLimit', 60) # Add a time limit in seconds to prevent very long optimization runs

                # Define weight variables for each asset in the optimization
                # Use the correctly scoped assets_for_optimization here
                weights = model.addVars(assets_for_optimization, name="w", lower=0.0) # Weights must be non-negative (long-only)

                # Set objective: Mean-Variance Optimization
                # Maximize Expected Return - 0.5 * gamma * Variance
                # The objective function for Gurobi (which minimizes) is:
                # minimize 0.5 * gamma * w' * Sigma * w - mu' * w

                # Quadratic term: 0.5 * gamma * w' * Sigma * w (Variance component)
                quadratic_term = gp.quicksum(cov_matrix.loc[a, b] * weights[a] * weights[b]
                                            for a in assets_for_optimization for b in assets_for_optimization)

                # Linear term: mu' * w (Expected Return component)
                linear_term = gp.quicksum(expected_returns[asset] * weights[asset]
                                          for asset in assets_for_optimization)

                # The objective to minimize: 0.5 * gamma * Variance - Expected Return
                objective = 0.5 * self.gamma * quadratic_term - linear_term

                model.setObjective(objective, gp.GRB.MINIMIZE)

                # Add constraint: The sum of portfolio weights must equal 1
                model.addConstr(gp.quicksum(weights[a] for a in assets_for_optimization) == 1, "SumToOne")

                # Optimize the model to find the optimal weights
                model.optimize()

                # Store the optimal weights if optimization was successful
                if model.status == gp.GRB.OPTIMAL:
                    for asset in assets_for_optimization:
                        self.portfolio_weights.loc[current_date, asset] = weights[asset].X
                    # SPY weight is already set to 0 for this date outside the loop
                else:
                    # If optimization fails for any reason, print a warning
                    print(f"Warning: Gurobi did not find an optimal solution on {current_date}. Status: {model.status}")
                    # The weights for this day will remain NaN, which will be handled by ffill later.

            except gp.GurobiError as e:
                # Catch and report Gurobi-specific errors
                print(f"Gurobi error on {current_date}: {e}")
            except Exception as e:
                # Catch and report any other unexpected errors during optimization
                print(f"An unexpected error occurred on {current_date}: {e}")


        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


"""
Assignment Judge

The following functions will help check your solution.
"""


class AssignmentJudge:
    def __init__(self):
        self.mp = MyPortfolio(df, "SPY").get_results()
        self.Bmp = MyPortfolio(Bdf, "SPY").get_results()

    def plot_performance(self, price, strategy):
        # Plot cumulative returns
        _, ax = plt.subplots()
        returns = price.pct_change().fillna(0)
        (1 + returns["SPY"]).cumprod().plot(ax=ax, label="SPY")
        (1 + strategy[1]["Portfolio"]).cumprod().plot(ax=ax, label=f"MyPortfolio")

        ax.set_title("Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        plt.show()
        return None

    def plot_allocation(self, df_weights):
        df_weights = df_weights.fillna(0).ffill()

        # long only
        df_weights[df_weights < 0] = 0

        # Plotting
        _, ax = plt.subplots()
        df_weights.plot.area(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Allocation")
        ax.set_title("Asset Allocation Over Time")
        plt.show()
        return None

    def report_metrics(self, price, strategy, show=False):
        df_bl = pd.DataFrame()
        returns = price.pct_change().fillna(0)
        df_bl["SPY"] = returns["SPY"]
        df_bl["MP"] = pd.to_numeric(strategy[1]["Portfolio"], errors="coerce")
        sharpe_ratio = qs.stats.sharpe(df_bl)

        if show == True:
            qs.reports.metrics(df_bl, mode="full", display=show)

        return sharpe_ratio

    def cumulative_product(self, dataframe):
        (1 + dataframe.pct_change().fillna(0)).cumprod().plot()

    def check_sharp_ratio_greater_than_one(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if self.report_metrics(df, self.mp)[1] > 1:
            print("Problem 4.1 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.1 Fail")
        return 0

    def check_sharp_ratio_greater_than_spy(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if (
            self.report_metrics(Bdf, self.Bmp)[1]
            > self.report_metrics(Bdf, self.Bmp)[0]
        ):
            print("Problem 4.2 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.2 Fail")
        return 0

    def check_portfolio_position(self, portfolio_weights):
        if (portfolio_weights.sum(axis=1) <= 1.01).all():
            return True
        print("Portfolio Position Exceeds 1. No Leverage.")
        return False

    def check_all_answer(self):
        score = 0
        score += self.check_sharp_ratio_greater_than_one()
        score += self.check_sharp_ratio_greater_than_spy()
        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()

    if args.score:
        if ("one" in args.score) or ("spy" in args.score):
            if "one" in args.score:
                judge.check_sharp_ratio_greater_than_one()
            if "spy" in args.score:
                judge.check_sharp_ratio_greater_than_spy()
        elif "all" in args.score:
            print(f"==> total Score = {judge.check_all_answer()} <==")

    if args.allocation:
        if "mp" in args.allocation:
            judge.plot_allocation(judge.mp[0])
        if "bmp" in args.allocation:
            judge.plot_allocation(judge.Bmp[0])

    if args.performance:
        if "mp" in args.performance:
            judge.plot_performance(df, judge.mp)
        if "bmp" in args.performance:
            judge.plot_performance(Bdf, judge.Bmp)

    if args.report:
        if "mp" in args.report:
            judge.report_metrics(df, judge.mp, show=True)
        if "bmp" in args.report:
            judge.report_metrics(Bdf, judge.Bmp, show=True)

    if args.cumulative:
        if "mp" in args.cumulative:
            judge.cumulative_product(df)
        if "bmp" in args.cumulative:
            judge.cumulative_product(Bdf)
