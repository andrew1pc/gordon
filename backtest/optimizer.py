"""Parameter optimization framework for strategy tuning."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from itertools import product
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics


logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """
    Parameter optimization framework for strategy tuning.

    Supports grid search, random search, and walk-forward optimization
    to find robust parameter sets.

    Example:
        >>> optimizer = ParameterOptimizer(
        ...     optimization_range=('2022-01-01', '2023-12-31'),
        ...     validation_range=('2024-01-01', '2024-06-30')
        ... )
        >>> results = optimizer.grid_search(param_grid)
    """

    def __init__(
        self,
        optimization_range: Tuple[str, str],
        validation_range: Optional[Tuple[str, str]] = None,
        optimization_metric: str = 'sharpe_ratio',
        n_jobs: int = 1
    ):
        """
        Initialize parameter optimizer.

        Args:
            optimization_range: (start_date, end_date) for optimization
            validation_range: Optional (start_date, end_date) for validation
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            n_jobs: Number of parallel workers (default 1)
        """
        self.optimization_range = optimization_range
        self.validation_range = validation_range
        self.optimization_metric = optimization_metric
        self.n_jobs = n_jobs

        # Results storage
        self.results = []
        self.cache_dir = 'cache/optimization'
        os.makedirs(self.cache_dir, exist_ok=True)

    def define_parameter_grid(self, params_dict: Dict[str, List]) -> List[Dict]:
        """
        Define parameter grid for optimization.

        Args:
            params_dict: Dict mapping parameter names to lists of values

        Returns:
            List of parameter combinations

        Example:
            >>> param_grid = {
            ...     'momentum_threshold': [65, 70, 75],
            ...     'risk_per_trade': [0.008, 0.010, 0.012]
            ... }
            >>> combinations = optimizer.define_parameter_grid(param_grid)
        """
        param_names = list(params_dict.keys())
        param_values = list(params_dict.values())

        # Generate all combinations
        combinations = list(product(*param_values))
        total_combinations = len(combinations)

        logger.info(f"Parameter grid: {total_combinations} combinations")

        # Warn if too many
        if total_combinations > 1000:
            logger.warning(
                f"Large parameter grid ({total_combinations} combinations). "
                "Consider using random_search instead."
            )

        # Convert to list of dicts
        param_list = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            param_list.append(param_dict)

        return param_list

    def grid_search(
        self,
        param_grid: Dict[str, List],
        max_combinations: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Exhaustive grid search over parameter space.

        Args:
            param_grid: Dict defining parameter ranges
            max_combinations: Optional limit on combinations to test

        Returns:
            DataFrame with results sorted by optimization metric

        Example:
            >>> results = optimizer.grid_search({
            ...     'momentum_threshold': [65, 70, 75],
            ...     'rsi_min': [45, 50, 55]
            ... })
        """
        logger.info("Starting grid search optimization...")

        # Generate parameter combinations
        param_list = self.define_parameter_grid(param_grid)

        # Limit if requested
        if max_combinations and len(param_list) > max_combinations:
            logger.info(f"Limiting to {max_combinations} combinations")
            param_list = param_list[:max_combinations]

        # Evaluate each combination
        results = []
        for i, params in enumerate(param_list):
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Testing {i+1}/{len(param_list)} combinations...")

            result = self.evaluate_parameters(params)
            if result:
                results.append(result)

        # Convert to DataFrame and rank
        results_df = pd.DataFrame(results)
        ranked = self.rank_results(results_df)

        logger.info(f"Grid search complete: tested {len(param_list)} combinations")
        return ranked

    def random_search(
        self,
        param_distributions: Dict[str, Tuple],
        n_iterations: int = 100
    ) -> pd.DataFrame:
        """
        Random search over parameter space.

        Args:
            param_distributions: Dict mapping param name to (min, max) range
            n_iterations: Number of random samples to test

        Returns:
            DataFrame with results

        Example:
            >>> results = optimizer.random_search({
            ...     'momentum_threshold': (60, 80),
            ...     'risk_per_trade': (0.005, 0.015)
            ... }, n_iterations=50)
        """
        logger.info(f"Starting random search: {n_iterations} iterations...")

        results = []
        for i in range(n_iterations):
            if (i + 1) % 10 == 0:
                logger.info(f"Testing {i+1}/{n_iterations}...")

            # Sample random parameters
            params = {}
            for param_name, (min_val, max_val) in param_distributions.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)

            result = self.evaluate_parameters(params)
            if result:
                results.append(result)

        results_df = pd.DataFrame(results)
        ranked = self.rank_results(results_df)

        logger.info(f"Random search complete: {len(results)} successful evaluations")
        return ranked

    def walk_forward_optimization(
        self,
        param_grid: Dict[str, List],
        n_splits: int = 5,
        train_pct: float = 0.8
    ) -> Dict:
        """
        Walk-forward optimization to prevent overfitting.

        Args:
            param_grid: Parameter grid to optimize
            n_splits: Number of time periods
            train_pct: Fraction of each period for training (0.8 = 80%)

        Returns:
            Dict with walk-forward results

        Example:
            >>> wf_results = optimizer.walk_forward_optimization(
            ...     param_grid, n_splits=5
            ... )
        """
        logger.info(f"Starting walk-forward optimization: {n_splits} splits...")

        # Parse date range
        start_date = pd.to_datetime(self.optimization_range[0])
        end_date = pd.to_datetime(self.optimization_range[1])
        total_days = (end_date - start_date).days

        # Calculate split size
        split_size = total_days // n_splits

        wf_results = {
            'splits': [],
            'in_sample_performance': [],
            'out_of_sample_performance': [],
            'best_params_per_split': [],
            'overall_best_params': None
        }

        # Process each split
        for split_idx in range(n_splits):
            split_start = start_date + timedelta(days=split_idx * split_size)
            split_end = split_start + timedelta(days=split_size)

            # Training period (first 80%)
            train_days = int(split_size * train_pct)
            train_start = split_start
            train_end = split_start + timedelta(days=train_days)

            # Testing period (last 20%)
            test_start = train_end
            test_end = split_end

            logger.info(
                f"Split {split_idx + 1}/{n_splits}: "
                f"Train {train_start.date()} to {train_end.date()}, "
                f"Test {test_start.date()} to {test_end.date()}"
            )

            # Optimize on training period
            temp_optimizer = ParameterOptimizer(
                optimization_range=(train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d')),
                optimization_metric=self.optimization_metric
            )
            train_results = temp_optimizer.grid_search(param_grid, max_combinations=50)

            if train_results.empty:
                logger.warning(f"No results for split {split_idx + 1}")
                continue

            # Best parameters from training
            best_params = train_results.iloc[0]['parameters']
            best_is_metric = train_results.iloc[0][self.optimization_metric]

            # Test on testing period
            test_result = self._evaluate_on_period(
                best_params,
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d')
            )

            if test_result:
                oos_metric = test_result.get(self.optimization_metric, 0)

                wf_results['splits'].append({
                    'split_idx': split_idx,
                    'train_period': (train_start.date(), train_end.date()),
                    'test_period': (test_start.date(), test_end.date())
                })
                wf_results['in_sample_performance'].append(best_is_metric)
                wf_results['out_of_sample_performance'].append(oos_metric)
                wf_results['best_params_per_split'].append(best_params)

                logger.info(
                    f"Split {split_idx + 1}: IS {self.optimization_metric}={best_is_metric:.3f}, "
                    f"OOS {self.optimization_metric}={oos_metric:.3f}"
                )

        # Find parameters with best average OOS performance
        if wf_results['out_of_sample_performance']:
            best_oos_idx = np.argmax(wf_results['out_of_sample_performance'])
            wf_results['overall_best_params'] = wf_results['best_params_per_split'][best_oos_idx]

            avg_oos = np.mean(wf_results['out_of_sample_performance'])
            logger.info(f"Walk-forward complete: Average OOS {self.optimization_metric} = {avg_oos:.3f}")

        return wf_results

    def evaluate_parameters(self, params: Dict) -> Optional[Dict]:
        """
        Evaluate single parameter set.

        Args:
            params: Parameter dict

        Returns:
            Dict with results or None if failed

        Example:
            >>> result = optimizer.evaluate_parameters({
            ...     'momentum_threshold': 70,
            ...     'risk_per_trade': 0.01
            ... })
        """
        try:
            # This is a simplified version - in production, would create
            # BacktestEngine with these parameters and run backtest
            # For now, return mock results

            # Mock implementation - replace with actual backtest
            mock_sharpe = np.random.uniform(0.5, 2.0)
            mock_return = np.random.uniform(-0.1, 0.5)
            mock_drawdown = np.random.uniform(0.05, 0.3)

            return {
                'parameters': params,
                'sharpe_ratio': mock_sharpe,
                'total_return': mock_return,
                'max_drawdown': mock_drawdown,
                'total_trades': np.random.randint(20, 100),
                'win_rate': np.random.uniform(0.4, 0.7)
            }

        except Exception as e:
            logger.error(f"Failed to evaluate parameters {params}: {e}")
            return None

    def _evaluate_on_period(
        self,
        params: Dict,
        start_date: str,
        end_date: str
    ) -> Optional[Dict]:
        """Evaluate parameters on specific period."""
        # Similar to evaluate_parameters but for specific date range
        try:
            mock_sharpe = np.random.uniform(0.3, 1.5)
            mock_return = np.random.uniform(-0.2, 0.4)

            return {
                'parameters': params,
                'sharpe_ratio': mock_sharpe,
                'total_return': mock_return,
                'period': (start_date, end_date)
            }
        except:
            return None

    def rank_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank optimization results.

        Args:
            results_df: DataFrame with optimization results

        Returns:
            Ranked DataFrame

        Example:
            >>> ranked = optimizer.rank_results(results_df)
        """
        if results_df.empty:
            return results_df

        # Apply filters
        filtered = results_df.copy()

        # Filter: minimum trades
        if 'total_trades' in filtered.columns:
            filtered = filtered[filtered['total_trades'] >= 20]

        # Filter: maximum drawdown
        if 'max_drawdown' in filtered.columns:
            filtered = filtered[filtered['max_drawdown'] <= 0.30]

        # Filter: minimum win rate
        if 'win_rate' in filtered.columns:
            filtered = filtered[filtered['win_rate'] >= 0.40]

        # Sort by optimization metric
        if self.optimization_metric in filtered.columns:
            filtered = filtered.sort_values(self.optimization_metric, ascending=False)

        logger.info(f"Ranking: {len(filtered)} results after filtering (from {len(results_df)})")

        return filtered.reset_index(drop=True)

    def analyze_parameter_sensitivity(
        self,
        results_df: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Analyze parameter sensitivity.

        Args:
            results_df: Results DataFrame

        Returns:
            Dict with sensitivity analysis

        Example:
            >>> sensitivity = optimizer.analyze_parameter_sensitivity(results)
        """
        if results_df.empty or 'parameters' not in results_df.columns:
            return {}

        sensitivity = {}

        # Extract all parameter names
        first_params = results_df.iloc[0]['parameters']
        param_names = list(first_params.keys())

        for param_name in param_names:
            # Extract parameter values and corresponding metrics
            param_values = []
            metric_values = []

            for _, row in results_df.iterrows():
                param_val = row['parameters'].get(param_name)
                metric_val = row.get(self.optimization_metric)

                if param_val is not None and metric_val is not None:
                    param_values.append(param_val)
                    metric_values.append(metric_val)

            if len(param_values) > 1:
                # Calculate sensitivity (variance in metric)
                metric_std = np.std(metric_values)
                metric_range = np.max(metric_values) - np.min(metric_values)

                sensitivity[param_name] = {
                    'std': metric_std,
                    'range': metric_range,
                    'mean_metric': np.mean(metric_values),
                    'robust': metric_std < 0.3  # Low variance = robust
                }

        return sensitivity

    def detect_overfitting(
        self,
        in_sample_metric: float,
        out_of_sample_metric: float
    ) -> Dict:
        """
        Detect overfitting by comparing IS vs OOS performance.

        Args:
            in_sample_metric: In-sample performance
            out_of_sample_metric: Out-of-sample performance

        Returns:
            Dict with overfitting analysis

        Example:
            >>> analysis = optimizer.detect_overfitting(2.5, 0.8)
        """
        if in_sample_metric <= 0:
            return {'overfitted': True, 'score': 100, 'message': 'Invalid in-sample metric'}

        # Calculate degradation
        degradation = (in_sample_metric - out_of_sample_metric) / in_sample_metric

        # Overfitting score (0-100, higher = more overfitting)
        overfitting_score = min(100, max(0, degradation * 100))

        # Determine if overfitted
        overfitted = degradation > 0.30  # >30% degradation

        message = "No significant overfitting detected"
        if overfitted:
            message = f"WARNING: Significant overfitting detected ({degradation:.1%} degradation)"

        return {
            'overfitted': overfitted,
            'score': overfitting_score,
            'degradation': degradation,
            'in_sample': in_sample_metric,
            'out_of_sample': out_of_sample_metric,
            'message': message
        }

    def generate_optimization_report(
        self,
        results_df: pd.DataFrame,
        output_file: str = 'output/optimization/optimization_report.txt'
    ) -> str:
        """
        Generate comprehensive optimization report.

        Args:
            results_df: Optimization results
            output_file: Path to save report

        Returns:
            Report as string

        Example:
            >>> report = optimizer.generate_optimization_report(results)
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PARAMETER OPTIMIZATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Optimization summary
        lines.append("OPTIMIZATION SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Optimization Period: {self.optimization_range[0]} to {self.optimization_range[1]}")
        lines.append(f"Optimization Metric: {self.optimization_metric}")
        lines.append(f"Total Combinations Tested: {len(results_df)}")
        lines.append("")

        # Top parameter sets
        if not results_df.empty:
            lines.append("TOP 10 PARAMETER SETS")
            lines.append("-" * 80)

            top_10 = results_df.head(10)
            for idx, row in top_10.iterrows():
                lines.append(f"\nRank #{idx + 1}:")
                lines.append(f"  {self.optimization_metric}: {row[self.optimization_metric]:.3f}")
                lines.append(f"  Parameters: {row['parameters']}")
                if 'total_trades' in row:
                    lines.append(f"  Trades: {row['total_trades']}")
                if 'win_rate' in row:
                    lines.append(f"  Win Rate: {row['win_rate']:.1%}")

            lines.append("")

            # Recommended configuration
            lines.append("RECOMMENDED CONFIGURATION")
            lines.append("-" * 80)
            best = results_df.iloc[0]
            lines.append(f"Parameters: {best['parameters']}")
            lines.append(f"Expected {self.optimization_metric}: {best[self.optimization_metric]:.3f}")
            lines.append("")

        lines.append("=" * 80)

        report = "\n".join(lines)

        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(report)

        logger.info(f"Optimization report saved to {output_file}")
        return report

    def plot_optimization_results(
        self,
        results_df: pd.DataFrame,
        output_dir: str = 'output/optimization'
    ) -> None:
        """
        Create optimization visualization.

        Args:
            results_df: Results DataFrame
            output_dir: Directory to save plots

        Example:
            >>> optimizer.plot_optimization_results(results)
        """
        if results_df.empty:
            logger.warning("No results to plot")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Plot 1: Metric distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        if self.optimization_metric in results_df.columns:
            ax.hist(results_df[self.optimization_metric], bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(results_df[self.optimization_metric].mean(),
                      color='red', linestyle='--', label='Mean')
            ax.set_xlabel(self.optimization_metric.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {self.optimization_metric.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/metric_distribution.png', dpi=300)
            plt.close()

        logger.info(f"Optimization plots saved to {output_dir}/")
