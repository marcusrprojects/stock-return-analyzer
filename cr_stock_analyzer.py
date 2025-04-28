# cr_stock_analyzer.py

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import numpy as np
import pandas as pd
from datetime import datetime
import os
import argparse
from typing import Dict, Tuple, Optional, Any

# --- Configuration ---

# Default values (can be overridden by command-line args)
DEFAULT_TICKER = "DIS"
DEFAULT_BENCHMARK = "^GSPC"
DEFAULT_OUTPUT_DIR = "data_viz"

# Time scopes in approximate trading days
SCOPES: Dict[str, int] = {
    '1 Month': 21,
    '3 Months': 63,
    '6 Months': 126,
    '1 Year': 252,
    '5 Years': 1260,
    '10 Years': 2520
}

# --- Core Functions ---

def get_stock_data(ticker: str, period: str = "max") -> pd.DataFrame:
    """
    Retrieves historical stock data from Yahoo Finance and calculates normalized 'Close' price.

    Args:
        ticker (str): The stock ticker symbol (e.g., "DIS").
        period (str): The period for which to fetch data (yf.Ticker valid period). Defaults to "max".

    Returns:
        pd.DataFrame: DataFrame containing historical stock data with 'Normalized' column.
                      Returns an empty DataFrame if fetching fails or data is invalid.
    """
    print(f"Fetching data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)

        if data.empty:
            print(f"Warning: No data returned for ticker {ticker} and period {period}.")
            return pd.DataFrame()

        if 'Close' not in data.columns or data['Close'].isnull().all() or len(data) == 0:
            print(f"Warning: 'Close' data missing or invalid for ticker {ticker}.")
            data['Normalized'] = pd.NA
            return data # Return with NA column if needed downstream, or empty

        # Ensure first close price is valid for normalization
        first_close = data['Close'].iloc[0]
        if pd.isna(first_close) or first_close == 0:
            print(f"Warning: Invalid first 'Close' price ({first_close}) for normalization of {ticker}.")
            data['Normalized'] = pd.NA
        else:
            # Normalize based on the very first closing price in the fetched data
            data['Normalized'] = data['Close'] / first_close

        print(f"Successfully fetched data for {ticker}.")
        return data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def align_data(stock_data: pd.DataFrame, benchmark_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns two DataFrames to their common index dates.

    Args:
        stock_data (pd.DataFrame): DataFrame for the primary stock.
        benchmark_data (pd.DataFrame): DataFrame for the benchmark.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Aligned stock_data and benchmark_data.
                                            Returns empty DataFrames if alignment fails or inputs are empty.
    """
    if stock_data.empty or benchmark_data.empty:
        print("Warning: Cannot align empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        common_dates = stock_data.index.intersection(benchmark_data.index)
        if common_dates.empty:
            print("Warning: No common dates found between stock and benchmark data.")
            return pd.DataFrame(), pd.DataFrame()

        aligned_stock = stock_data.loc[common_dates]
        aligned_benchmark = benchmark_data.loc[common_dates]
        print(f"Data aligned to {len(common_dates)} common dates.")
        return aligned_stock, aligned_benchmark
    except Exception as e:
        print(f"Error aligning data: {e}")
        return pd.DataFrame(), pd.DataFrame()


def calculate_cumulative_return_from_start(normalized_series: pd.Series) -> pd.Series:
    """
    Calculates cumulative percentage return relative to the first value in the series.
    Assumes input series is already normalized relative to the overall data start.

    Args:
        normalized_series (pd.Series): A Series of normalized prices (e.g., price / first_price).

    Returns:
        pd.Series: Series representing the cumulative return percentage relative to the series start.
                   Returns empty Series on error or invalid input.
    """
    if normalized_series.empty or normalized_series.isnull().all():
        print("Warning: Cannot calculate cumulative return on empty or all-NaN series.")
        return pd.Series(dtype=float)
    try:
        # Calculate cumulative return from the start of *this specific scope*
        first_val = normalized_series.iloc[0]
        if pd.isna(first_val) or first_val == 0:
            print(f"Warning: Invalid first value ({first_val}) for cumulative return calculation within scope.")
            # Return NaN series to indicate calculation failure for this scope
            return pd.Series(np.nan, index=normalized_series.index)
        return (normalized_series / first_val) - 1
    except IndexError:
        print("Warning: Cannot calculate cumulative return on empty series (IndexError).")
        return pd.Series(dtype=float)
    except Exception as e:
        print(f"Error calculating cumulative return: {e}")
        return pd.Series(dtype=float)

# --- Plotting Functions ---

def _set_xticks_by_period(ax: plt.Axes, date_index: pd.DatetimeIndex, period_days: int, text_color: str = 'black'):
    """Sets appropriate x-axis ticks and labels based on the time period length."""
    if date_index.empty: return

    # Thresholds slightly above exact number of days for better interval grouping
    if period_days <= 21 * 1.5:  # ~1.5 Months -> Weekly ticks
        locator = mdates.WeekdayLocator(interval=1)
        formatter = mdates.DateFormatter('%Y-%m-%d')
    elif period_days <= 126 * 1.5: # ~9 Months -> Monthly ticks
        locator = mdates.MonthLocator(interval=1)
        formatter = mdates.DateFormatter('%Y-%m')
    elif period_days <= 252 * 1.5: # ~1.5 Years -> Bi-monthly ticks
        locator = mdates.MonthLocator(interval=2)
        formatter = mdates.DateFormatter('%Y-%m')
    elif period_days <= 1260 * 1.5: # ~7.5 Years -> Yearly ticks
        locator = mdates.YearLocator()
        formatter = mdates.DateFormatter('%Y')
    else: # > 7.5 Years -> 2-Year ticks
        locator = mdates.YearLocator(base=2)
        formatter = mdates.DateFormatter('%Y')

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', rotation=45, colors=text_color)
    ax.tick_params(axis='y', colors=text_color)

def _prepare_scope_data(norm_stock_series: pd.Series, norm_bench_series: pd.Series, period_days: int, smooth_window: Optional[int] = None) -> Tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    """ Prepares data (slicing, cumulative return, smoothing) for a single plot scope. """
    if norm_stock_series.empty or norm_bench_series.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DatetimeIndex([])

    # Slice data for the scope (ensure enough data exists)
    if len(norm_stock_series) < period_days:
        print(f"Warning: Not enough data ({len(norm_stock_series)} days) for scope ({period_days} days). Using available data.")
        period_days = len(norm_stock_series) # Adjust period to available data

    stock_scope = norm_stock_series.iloc[-period_days:]
    bench_scope = norm_bench_series.iloc[-period_days:]

    # Calculate cumulative returns for this specific scope
    stock_cum_return = calculate_cumulative_return_from_start(stock_scope)
    bench_cum_return = calculate_cumulative_return_from_start(bench_scope)

    # Handle potential NaN series from calculation failure
    if stock_cum_return.isnull().all() or bench_cum_return.isnull().all():
        print(f"Warning: Cumulative return calculation failed for scope {period_days} days.")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DatetimeIndex([])


    # Apply smoothing if window is provided
    if smooth_window and smooth_window > 1:
        stock_cum_return = stock_cum_return.rolling(window=smooth_window, min_periods=1).mean()
        bench_cum_return = bench_cum_return.rolling(window=smooth_window, min_periods=1).mean()

    # Get index for plotting (use stock's index, should be same as benchmark after alignment)
    date_index = stock_scope.index

    return stock_cum_return, bench_cum_return, date_index


def plot_cumulative_returns(
    norm_stock_data: pd.Series,
    norm_bench_data: pd.Series,
    scopes: Dict[str, int],
    ticker: str,
    benchmark_ticker: str,
    plot_options: Dict[str, Any]
):
    """
    Generates a 2x3 grid of plots showing cumulative returns for different scopes.

    Args:
        norm_stock_data (pd.Series): Aligned and normalized stock data ('Normalized' column).
        norm_bench_data (pd.Series): Aligned and normalized benchmark data ('Normalized' column).
        scopes (Dict[str, int]): Dictionary mapping scope labels to period lengths (days).
        ticker (str): The stock ticker symbol.
        benchmark_ticker (str): The benchmark ticker symbol.
        plot_options (Dict[str, Any]): Dictionary containing plot styling parameters.
    """
    if norm_stock_data.empty or norm_bench_data.empty:
        print("Error: Cannot plot with empty data series.")
        return

    # --- Plot Setup ---
    try:
        # Attempt to set font properties, provide fallback
        font_properties = font_manager.FontProperties(family=plot_options.get('font_family', 'sans-serif'))
        plt.rc('font', family=plot_options.get('font_family', 'sans-serif'))
    except Exception as e:
        print(f"Warning: Could not set font family '{plot_options.get('font_family')}'. Using default. Error: {e}")
        font_properties = font_manager.FontProperties(family='sans-serif') # Fallback
        plt.rc('font', family='sans-serif')


    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=plot_options['background_color'])
    fig.suptitle(f'Cumulative Returns of {ticker} vs. {benchmark_ticker}',
                 fontsize=28, fontproperties=font_properties,
                 color=plot_options['text_color'], y=1.02)

    legend_handles = []
    legend_labels = []

    # Define base window size and increment for progressive smoothing
    base_window = 5
    window_increment = 5
    smoothing_start_index = 3 # Start smoothing from the Nth plot (0-indexed)

    # --- Plotting Loop ---
    for i, (label, period_days) in enumerate(scopes.items()):
        ax = axes[i // 3, i % 3]
        ax.set_facecolor(plot_options['graph_bg_color'])

        # Determine smoothing window
        smooth_window = None
        if i >= smoothing_start_index:
            # Progressive smoothing: window increases for later charts (index >= 3)
            smooth_window = base_window + window_increment * (i - smoothing_start_index + 1)

        # Prepare data for this scope
        stock_ret, bench_ret, dates = _prepare_scope_data(
            norm_stock_data, norm_bench_data, period_days, smooth_window
        )

        if stock_ret.empty or bench_ret.empty or dates.empty:
            print(f"Warning: Skipping scope '{label}' due to insufficient data after processing.")
            ax.set_title(f'{label}\n(Insufficient Data)', fontproperties=font_properties, color='red', fontsize=14)
            ax.set_xticks([]) # Remove ticks if no data
            ax.set_yticks([])
            continue # Skip to next scope if data is bad

        # --- Plotting Scope ---
        try:
            # Remove NaNs after smoothing for correct limit setting if tightXLim is true
            valid_stock_ret = stock_ret.dropna()
            non_nan_dates = valid_stock_ret.index

            stock_line, = ax.plot(dates, stock_ret, label=ticker,
                                color=plot_options['stock_color'],
                                linestyle=plot_options['linestyle'])
            bench_line, = ax.plot(dates, bench_ret, label=benchmark_ticker,
                                color=plot_options['bench_color'],
                                linestyle=plot_options['linestyle'])

            # Collect legend handles only once
            if i == 0:
                legend_handles.extend([stock_line, bench_line])
                legend_labels.extend([ticker, benchmark_ticker])

            # Fill between curves
            if plot_options['fill']:
                ax.fill_between(dates, stock_ret, bench_ret,
                                where=(stock_ret >= bench_ret),
                                facecolor=plot_options['stock_color'], alpha=0.3, interpolate=True)
                ax.fill_between(dates, stock_ret, bench_ret,
                                where=(bench_ret > stock_ret),
                                facecolor=plot_options['bench_color'], alpha=0.3, interpolate=True)

            # --- Formatting Scope ---
            ax.set_title(f'{label}', fontproperties=font_properties, color=plot_options['text_color'], fontsize=16)
            ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format)) # Format Y axis as percentage

            _set_xticks_by_period(ax, dates, period_days, plot_options['text_color'])

            # Set x-axis limits tightly to data if requested
            if plot_options['tight_xlim'] and not non_nan_dates.empty:
                 # Add a small padding to prevent cutting off lines at edges
                 date_padding = pd.Timedelta(days=max(1, period_days * 0.01))
                 ax.set_xlim(non_nan_dates.min() - date_padding, non_nan_dates.max() + date_padding)
            elif not dates.empty:
                 ax.set_xlim(dates.min(), dates.max()) # Default limits

            # Labels, Ticks, Grid
            if plot_options['show_labels']:
                ax.set_xlabel('Date', fontproperties=font_properties, color=plot_options['text_color'])
                ax.set_ylabel('Cumulative Return', fontproperties=font_properties, color=plot_options['text_color'])
            else:
                ax.set_xlabel('')
                ax.set_ylabel('')

            if plot_options['show_grid']:
                ax.grid(True, color=plot_options['text_color'], alpha=0.3, linestyle=':') # Lighter grid
            else:
                ax.grid(False)

            if not plot_options['show_ticks']:
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        except Exception as plot_err:
            print(f"Error plotting scope '{label}': {plot_err}")
            ax.set_title(f'{label}\n(Plotting Error)', fontproperties=font_properties, color='red', fontsize=14)


    # --- Final Touches ---
    # Add legend below the plots
    if plot_options['show_legend'] and legend_handles: # Ensure handles exist
        try:
            fig.legend(handles=legend_handles, labels=legend_labels,
                    loc='lower center',
                    fontsize=14, prop=font_properties,
                    frameon=True, framealpha=0.9, facecolor=plot_options['graph_bg_color'],
                    edgecolor=plot_options['text_color'], fancybox=True,
                    borderpad=0.8, ncol=2,
                    bbox_to_anchor=(0.5, -0.05)) # Adjust anchor slightly
        except Exception as legend_err:
            print(f"Warning: Could not display legend. Error: {legend_err}")


    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.97], h_pad=3, w_pad=3) # Adjust rect to avoid legend/title overlap
    except ValueError as layout_err:
        print(f"Warning: tight_layout encountered an issue: {layout_err}. Plot might have overlap.")


    # Save the plot
    if plot_options['save']:
        save_path = plot_options['save_path']
        output_dir = plot_options.get('output_dir', DEFAULT_OUTPUT_DIR) # Get output dir setting

        if not save_path:
            # Generate default filename
            date_str = datetime.now().strftime("%Y%m%d_%H%M")
            # Sanitize ticker names for filenames
            safe_ticker = "".join(c for c in ticker if c.isalnum() or c in ('-', '_')).rstrip()
            safe_benchmark = "".join(c for c in benchmark_ticker if c.isalnum() or c in ('-', '_')).rstrip()
            filename = f"{safe_ticker}-vs-{safe_benchmark}-CR-{date_str}.png"
            save_path = os.path.join(output_dir, filename)
        else:
            # Ensure directory exists if a full path is given in save_path
             output_dir = os.path.dirname(save_path)
             if not output_dir: # If only filename given in --output
                 output_dir = plot_options.get('output_dir', DEFAULT_OUTPUT_DIR) # Use default dir
                 save_path = os.path.join(output_dir, save_path)


        try:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.3, facecolor=fig.get_facecolor(), dpi=150)
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")

    plt.show()

# --- Main Execution ---

def main():
    """Main function to parse arguments, fetch data, and generate plots."""
    parser = argparse.ArgumentParser(description="Analyze and plot cumulative stock returns against a benchmark.")
    parser.add_argument("-t", "--ticker", type=str, default=DEFAULT_TICKER,
                        help=f"Stock ticker symbol (default: {DEFAULT_TICKER})")
    parser.add_argument("-b", "--benchmark", type=str, default=DEFAULT_BENCHMARK,
                        help=f"Benchmark ticker symbol (default: {DEFAULT_BENCHMARK})")
    parser.add_argument("-p", "--period", type=str, default="max",
                        help="Data fetch period (e.g., '5y', '10y', 'max') (default: max)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file path for the plot (e.g., 'plots/my_analysis.png'). "
                             "If not specified, a default filename is generated based on tickers and date.")
    parser.add_argument("--no-save", action="store_true",
                        help="Prevent saving the plot image.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Default directory for saving plots if --output is not set (default: {DEFAULT_OUTPUT_DIR})")

    # Plotting customization arguments
    parser.add_argument("--stock-color", type=str, default='#004d80', help="Color for the stock line (hex or name).")
    parser.add_argument("--bench-color", type=str, default='#80004d', help="Color for the benchmark line (hex or name).")
    parser.add_argument("--font-family", type=str, default='Times New Roman', help="Font family for plots (ensure it's available).")
    parser.add_argument("--bg-color", type=str, default='#FFF1E0', help="Figure background color.")
    parser.add_argument("--graph-bg-color", type=str, default='#F2DFCE', help="Axes background color.")
    parser.add_argument("--text-color", type=str, default='#322210', help="Text color for titles, labels, ticks.")
    parser.add_argument("--no-fill", action="store_true", help="Disable filling the area between curves.")
    parser.add_argument("--no-grid", action="store_true", help="Disable grid lines.")
    parser.add_argument("--no-ticks", action="store_true", help="Hide tick labels.")
    parser.add_argument("--show-labels", action="store_true", help="Show axis labels (default is hidden).")
    parser.add_argument("--no-legend", action="store_true", help="Hide the legend.")
    parser.add_argument("--tight-xlim", action="store_true", help="Set x-axis limits tightly to the data range (useful with smoothing).")

    args = parser.parse_args()

    # --- Data Fetching and Preparation ---
    stock_data_raw = get_stock_data(args.ticker, args.period)
    benchmark_data_raw = get_stock_data(args.benchmark, args.period)

    if stock_data_raw.empty or benchmark_data_raw.empty:
        print("Exiting due to data fetching errors.")
        return

    stock_data, benchmark_data = align_data(stock_data_raw, benchmark_data_raw)

    if stock_data.empty or benchmark_data.empty:
        print("Exiting due to data alignment errors or no common data.")
        return

    # Check if 'Normalized' columns exist and are valid
    if 'Normalized' not in stock_data.columns or 'Normalized' not in benchmark_data.columns or \
       stock_data['Normalized'].isnull().all() or benchmark_data['Normalized'].isnull().all():
        print("Exiting because 'Normalized' data series is missing or invalid after alignment.")
        return

    norm_stock_series = stock_data['Normalized']
    norm_bench_series = benchmark_data['Normalized']

    # --- Data Length Check ---
    min_scope_days = min(SCOPES.values()) if SCOPES else 0
    max_scope_days = max(SCOPES.values()) if SCOPES else 0

    if len(norm_stock_series) < min_scope_days and min_scope_days > 0:
         print(f"Error: Not enough aligned data ({len(norm_stock_series)} days) for the shortest scope ({min_scope_days} days). Cannot generate meaningful plots.")
         return
    elif len(norm_stock_series) < max_scope_days and max_scope_days > 0:
        print(f"Warning: Aligned data length ({len(norm_stock_series)} days) is less than the longest scope ({max_scope_days} days). Longer scopes will use available data.")

    # --- Plotting ---
    plot_options = {
        'stock_color': args.stock_color,
        'bench_color': args.bench_color,
        'font_family': args.font_family,
        'background_color': args.bg_color,
        'graph_bg_color': args.graph_bg_color,
        'text_color': args.text_color,
        'fill': not args.no_fill,
        'linestyle': '-',
        'show_grid': not args.no_grid,
        'show_ticks': not args.no_ticks,
        'show_labels': args.show_labels,
        'show_legend': not args.no_legend,
        'tight_xlim': args.tight_xlim,
        'save': not args.no_save,
        'save_path': args.output,
        'output_dir': args.output_dir
    }

    plot_cumulative_returns(
        norm_stock_series,
        norm_bench_series,
        SCOPES,
        args.ticker.upper(), # Ensure consistent ticker case
        args.benchmark.upper(), # Ensure consistent benchmark case
        plot_options
    )

if __name__ == "__main__":
    main()