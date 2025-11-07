# app.py
# OTE Strategy Dashboard v2.5 (Slider Fix)
#
# To run:
# 1. Make sure MT5 terminal is running (for Live Mode)
# 2. Open terminal in this folder
# 3. Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, time
import plotly.graph_objects as go
import plotly.express as px
import os
import matplotlib.pyplot as plt
import mplfinance as mpf

# --- **** THIS IS THE FIX **** ---
# Try to import MetaTrader5. If it fails (like on Streamlit Cloud),
# set a flag and continue.
try:
    import MetaTrader5 as mt5
    MT5_ENABLED = True
except (ImportError, ModuleNotFoundError): # Catch both errors
    print("MetaTrader5 library not found. Running in Offline-Only mode.")
    MT5_ENABLED = False
# --- **** END OF FIX **** ---


# -----------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------
st.set_page_config(
    page_title="OTE Strategy Dashboard",
    page_icon="⚡",
    layout="wide"
)

# -----------------------------------------------------------------
# CACHING - Speeds up the app
# -----------------------------------------------------------------

@st.cache_resource
def connect_mt5():
    """Initializes connection to MT5"""
    if not MT5_ENABLED:
        return False
    
    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return False
    print("MetaTrader 5 Initialized")
    return True

@st.cache_data
def get_symbol_info(symbol):
    """Gets symbol info and caches it."""
    print(f"Getting symbol info for {symbol}...")
    if not MT5_ENABLED or not connect_mt5():
        return None # Return None if MT5 isn't running or available

    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"Could not get symbol info for {symbol}")
        return None
        
    point = info.point
    if "XAU" in symbol or "XAG" in symbol:
        pip_size = 0.01
    elif symbol in ["BTCUSD"]:
         pip_size = 0.01
    elif symbol in ["ETHUSD"]:
         pip_size = 0.01
    elif info.digits == 5 or info.digits == 3:
        pip_size = point * 10
    else:
        pip_size = point

    # --- Determine Lot Value (Corrected) ---
    if symbol == "XAUUSD":
        lot_value_per_point = 100.0
    elif symbol == "ETHUSD":
        lot_value_per_point = 10.0
    elif symbol == "BTCUSD":
        lot_value_per_point = 1.0 # 1 lot = 1 coin, $1 move = $1 profit
    elif "USD" in symbol: # Forex
        lot_value_per_point = 100000.0
    else:
        # Fallback
        if info.trade_tick_value != 0 and info.trade_tick_size != 0:
            lot_value_per_point = info.trade_tick_value / info.trade_tick_size
        else:
             lot_value_per_point = 1.0
    
    # --- Determine Digits ---
    digits = info.digits
    if symbol == "ETHUSD":
        digits = 2 # Your value
        
    return {
        'pip_size': pip_size,
        'point': point,
        'lot_value_per_point': lot_value_per_point,
        'digits': digits
    }

# -----------------------------------------------------------------
# DATA LOADING FUNCTIONS
# -----------------------------------------------------------------

def fetch_mt5_data(symbol, timeframe, months):
    """Fetches live data from MT5."""
    if not MT5_ENABLED or not connect_mt5():
        st.error("Could not connect to MetaTrader 5. Please ensure the terminal is running.")
        return None, None

    tf_map = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}
    mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_M5)

    end = datetime.now(timezone.utc)
    total_months_to_fetch = months + 2
    start = end - timedelta(days=30 * total_months_to_fetch)
    test_start_date = end - timedelta(days=30 * months)
    
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, start, end)
    
    if rates is None or len(rates) == 0:
        st.error("No data fetched from MT5. Check symbol or MT5 connection.")
        return None, None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    df.set_index('time', inplace=True)
    
    return df, test_start_date

def load_offline_data(symbol, timeframe):
    """Loads data from a pre-exported CSV file."""
    # --- FIX: Look in current folder (./), not ./data/ ---
    filename = f"{symbol}_{timeframe}.csv" 
    if not os.path.exists(filename):
        st.error(f"Data file not found: {filename}. Please make sure it's in the same folder as app.py.")
        return None, None
        
    try:
        # Use delimiter=',' and encoding='latin1'
        print(f"Reading {filename} with delimiter=',' and encoding='latin1'")
        df = pd.read_csv(
            filename,
            delimiter=',',  # Your error showed comma delimited
            encoding='latin1' # Fixes unicode errors from MT5 export
        )
        
        # --- Fix column names (remove potential brackets/spaces and make UPPER) ---
        df.columns = df.columns.str.strip().str.replace('[<>]', '', regex=True).str.upper()

        # --- Use TIME column as timestamp, do not require DATE ---
        if 'TIME' not in df.columns:
            st.error(f"Error: CSV file {filename} is missing the 'TIME' column. Found columns: {list(df.columns)}")
            return None, None
            
        if 'DATE' in df.columns:
            print("Found DATE and TIME columns. Combining them.")
            df['time_str'] = df['DATE'] + ' ' + df['TIME']
        else:
            print("Found TIME column (no DATE column). Using TIME as timestamp.")
            df['time_str'] = df['TIME']
            
        df['time'] = pd.to_datetime(df['time_str'])
        df['time'] = df['time'].dt.tz_localize(timezone.utc)
        
        # --- Find the Volume column (TICK_VOLUME, TICKVOL, etc.) ---
        if 'TICK_VOLUME' in df.columns: # Your error log shows this column
            df.rename(columns={'TICK_VOLUME': 'volume'}, inplace=True)
        elif 'TICKVOL' in df.columns:
            df.rename(columns={'TICKVOL': 'volume'}, inplace=True)
        elif 'VOL' in df.columns:
            df.rename(columns={'VOL': 'volume'}, inplace=True)
        elif 'VOLUME' in df.columns:
            df.rename(columns={'VOLUME': 'volume'}, inplace=True)
        else:
            print(f"Warning: No Volume column found. Creating 'volume' column with 0s.")
            df['volume'] = 0
            
        # --- Rename OHLC columns ---
        df.rename(columns={
            'OPEN': 'open',
            'HIGH': 'high',
            'LOW': 'low',
            'CLOSE': 'close'
        }, inplace=True)
        
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            st.error(f"Error: CSV file {filename} is missing one or more OHLC columns (OPEN, HIGH, LOW, CLOSE).")
            return None, None
            
        df = df[required_cols]
        df.set_index('time', inplace=True)
        
        test_start_date = df.index[200] 
        return df, test_start_date
        
    except Exception as e:
        st.error(f"Error reading CSV file '{filename}': {e}")
        return None, None


# -----------------------------------------------------------------
# BACKTESTING ENGINE & HELPERS (Your Proven Logic)
# -----------------------------------------------------------------

def compute_atr( df, period=14 ):
    period = int(period)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def get_session_mask( df_index_eastern, start_time, end_time ):
    t = df_index_eastern.time
    if start_time <= end_time:
        return (t >= start_time) & (t <= end_time)
    else:
        return (t >= start_time) | (t <= end_time)


def calculate_daily_levels( df ):
    df_copy = df.copy()
    try:
        df_copy['time_eastern'] = df_copy.index.tz_convert('US/Eastern')
    except TypeError:
        df_copy['time_eastern'] = df_copy.index.tz_localize('UTC').tz_convert('US/Eastern')

    df_copy['date_eastern'] = df_copy['time_eastern'].dt.date

    asia_mask = get_session_mask(df_copy['time_eastern'].dt, time(20, 0), time(23, 59))
    asia_highs = df_copy[asia_mask].groupby('date_eastern')['high'].max()
    asia_lows = df_copy[asia_mask].groupby('date_eastern')['low'].min()
    df['asia_high'] = df_copy['date_eastern'].map(asia_highs).ffill().shift(1)
    df['asia_low'] = df_copy['date_eastern'].map(asia_lows).ffill().shift(1)

    daily_highs = df_copy.groupby('date_eastern')['high'].max()
    daily_lows = df_copy.groupby('date_eastern')['low'].min()
    df['pdh'] = df_copy['date_eastern'].map(daily_highs.shift(1)).ffill()
    df['pdl'] = df_copy['date_eastern'].map(daily_lows.shift(1)).ffill()

    return df


def in_session( bar_time_utc, windows ):
    try:
        eastern = bar_time_utc.tz_convert('US/Eastern')
    except Exception:
        eastern = bar_time_utc.tz_convert('US/Eastern')
    t = eastern.time()
    for s, e in windows:
        if s <= t <= e:
            return True
    return False


def lot_size_mt5( symbol_info, entry, sl, balance, risk_pct ):
    lot_value_per_point = symbol_info['lot_value_per_point']
    price_diff = abs(entry - sl)
    if price_diff <= 0:
        return 0.01
    risk_amount = balance * risk_pct
    value_per_lot = price_diff * lot_value_per_point
    lot = risk_amount / value_per_lot
    lot = max(round(float(lot), 2), 0.01)
    return lot


def run_backtest( df, params, symbol_info, test_start_date, require_fvg=False ):
    print("Running OTE backtest...")
    df = df.copy()
    df['ATR'] = compute_atr(df, params['atr_period'])
    df['EMA_TREND'] = df['close'].ewm(span=params['ema_trend_period'], adjust=False).mean()
    df = calculate_daily_levels(df)

    lot_value_per_point = symbol_info['lot_value_per_point']
    pip_size = symbol_info['pip_size']

    atr_sl_multiplier = float(params.get('atr_sl_multiplier', 1.5))
    min_swing_dist = float(params.get('min_swing_pips', 15.0)) * pip_size
    equity = float(params.get('initial_equity', 10000.0))
    trades = []
    lookback = int(params['lookback_swing'])
    rr = float(params['rr'])
    
    # Session Windows
    NY_SESSION = (time(7, 0), time(11, 0))
    LONDON_SESSION = (time(2, 0), time(6, 0))
    ENTRY_SESSIONS = [NY_SESSION, LONDON_SESSION]

    try:
        start_trade_idx = df.index.searchsorted(test_start_date, side='left')
    except KeyError:
        start_trade_idx = 0
        
    i = max(lookback, params['ema_trend_period'], start_trade_idx)
    
    if i >= len(df):
        print("Not enough data for warmup. Test start date is beyond data range.")
        return pd.DataFrame() # Return empty dataframe
        
    print(f"--- Indicators warmed up. Starting test from {df.index[i]} ---")

    trade_id_counter = 1
    swept_high_liquidity = False
    swept_low_liquidity = False
    current_day = None

    while i < len(df) - 1:
        bar_time = df.index[i]
        day = bar_time.date()
        if day != current_day:
            swept_high_liquidity, swept_low_liquidity = False, False
            current_day = day

        asia_high, asia_low = df['asia_high'].iat[i], df['asia_low'].iat[i]
        pdh, pdl = df['pdh'].iat[i], df['pdl'].iat[i]
        if pd.isna(asia_high) or pd.isna(pdh):
            i += 1; continue
        if df['high'].iat[i] > asia_high or df['high'].iat[i] > pdh:
            swept_high_liquidity = True
        if df['low'].iat[i] < asia_low or df['low'].iat[i] < pdl:
            swept_low_liquidity = True

        if not in_session(bar_time, ENTRY_SESSIONS):
            i += 1; continue
        current_atr = df['ATR'].iat[i]
        if current_atr == 0 or pd.isna(current_atr):
            i += 1; continue

        close_price = df['close'].iat[i]
        ema_trend = df['EMA_TREND'].iat[i]
        is_uptrend = close_price > ema_trend
        is_downtrend = close_price < ema_trend

        window = df.iloc[i - lookback: i]
        if window.empty:
            i += 1; continue

        swing_high_idx = window['high'].idxmax()
        swing_low_idx = window['low'].idxmin()
        swing_high = window.loc[swing_high_idx, 'high']
        swing_low = window.loc[swing_low_idx, 'low']
        swing_range = swing_high - swing_low

        if swing_range < min_swing_dist:
            i += 1; continue

        trade = None
        
        if is_uptrend and swept_low_liquidity and (swing_low_idx < swing_high_idx):
            ote_low_price = swing_low + swing_range * params['ote_low']
            ote_high_price = swing_low + swing_range * params['ote_high']
            if (ote_low_price <= close_price <= ote_high_price):
                entry_price = float(close_price)
                sl_price = float(swing_low - (current_atr * atr_sl_multiplier))
                tp_price = float(entry_price + (entry_price - sl_price) * rr)
                if sl_price < entry_price:
                    lot = lot_size_mt5(symbol_info, entry_price, sl_price, equity, params['risk_per_trade'])
                    trade = {'side': 'BUY', 'entry': entry_price, 'sl': sl_price, 'tp': tp_price, 'lot': lot,
                             'reason': f"Bullish OTE after Low Liq. Sweep (Trend UP)",
                             'swing_low': swing_low, 'swing_high': swing_high}

        elif is_downtrend and swept_high_liquidity and (swing_high_idx < swing_low_idx):
            ote_low_price = swing_low + swing_range * params['ote_low']
            ote_high_price = swing_low + swing_range * params['ote_high']
            if (ote_low_price <= close_price <= ote_high_price):
                entry_price = float(close_price)
                sl_price = float(swing_high + (current_atr * atr_sl_multiplier))
                tp_price = float(entry_price - (sl_price - entry_price) * rr)
                if sl_price > entry_price:
                    lot = lot_size_mt5(symbol_info, entry_price, sl_price, equity, params['risk_per_trade'])
                    trade = {'side': 'SELL', 'entry': entry_price, 'sl': sl_price, 'tp': tp_price, 'lot': lot,
                             'reason': f"Bearish OTE after High Liq. Sweep (Trend DOWN)",
                             'swing_low': swing_low, 'swing_high': swing_high}
        
        if trade is None:
            i += 1
            continue

        trade['id'] = trade_id_counter
        trade['entry_time'] = df.index[i]
        
        closed = False
        j = i + 1
        while j < len(df):
            row = df.iloc[j]
            pnl = 0.0
            exit_price = entry_price
            result = 'OPEN'
            if trade['side'] == 'BUY':
                if row['high'] >= tp_price:
                    exit_price, result = tp_price, 'WIN'
                elif row['low'] <= sl_price:
                    exit_price, result = sl_price, 'LOSS'
                pnl = (exit_price - entry_price) * lot * lot_value_per_point
            elif trade['side'] == 'SELL':
                if row['low'] <= tp_price:
                    exit_price, result = tp_price, 'WIN'
                elif row['high'] >= sl_price:
                    exit_price, result = sl_price, 'LOSS'
                pnl = (entry_price - exit_price) * lot * lot_value_per_point
            
            if result != 'OPEN':
                trade.update({'exit_time': df.index[j], 'exit': float(exit_price), 'result': result, 'pnl': float(pnl)})
                equity += pnl
                trades.append(trade)
                closed = True
                trade_id_counter += 1
                if trade['side'] == 'BUY': swept_low_liquidity = False
                else: swept_high_liquidity = False
                break
            j += 1
        if not closed:
            i = len(df)
            break
        i = j + 1

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.tz_localize(None)
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.tz_localize(None)
        trades_df.set_index('id', inplace=True)
    return trades_df

# -----------------------------------------------------------------
# ANALYTICS & PLOTTING FUNCTIONS
# -----------------------------------------------------------------

def calculate_insights(df_trades, initial_equity):
    """Calculates all advanced performance metrics."""
    if df_trades.empty:
        return {}
        
    insights = {}
    total_trades = len(df_trades)
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[df_trades['pnl'] < 0]
    total_wins = len(wins)
    total_losses = len(losses)
    win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    
    net_pnl = df_trades['pnl'].sum()
    gross_profit = wins['pnl'].sum()
    gross_loss = abs(losses['pnl'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999
    
    avg_win = wins['pnl'].mean() if total_wins > 0 else 0
    avg_loss = abs(losses['pnl'].mean()) if total_losses > 0 else 0
    expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
    
    df_trades['equity_curve'] = df_trades['pnl'].cumsum() + initial_equity
    df_trades['peak'] = df_trades['equity_curve'].cummax()
    df_trades['drawdown_usd'] = df_trades['equity_curve'] - df_trades['peak']
    max_drawdown_usd = df_trades['drawdown_usd'].min()
    
    df_trades['duration'] = (df_trades['exit_time'] - df_trades['entry_time'])
    avg_duration = df_trades['duration'].mean()
    
    side_performance = df_trades.groupby('side')['pnl'].sum()
    day_of_week_performance = df_trades.groupby(df_trades['entry_time'].dt.day_name())['pnl'].sum()
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_of_week_performance = day_of_week_performance.reindex(days_order).fillna(0)
    
    hour_of_day_performance = df_trades.groupby(df_trades['entry_time'].dt.hour)['pnl'].sum().sort_index()

    return {
        'net_pnl': net_pnl, 'total_trades': total_trades, 'win_rate': win_rate,
        'profit_factor': profit_factor, 'total_wins': total_wins, 'total_losses': total_losses,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'expectancy': expectancy,
        'max_drawdown_usd': max_drawdown_usd, 'avg_duration': str(avg_duration).split('.')[0],
        'side_performance': side_performance,
        'day_of_week_performance': day_of_week_performance,
        'hour_of_day_performance': hour_of_day_performance,
        'equity_curve': df_trades[['exit_time', 'equity_curve', 'peak']],
        'drawdown_curve': df_trades[['exit_time', 'drawdown_usd']],
        'pnl_distribution': df_trades['pnl']
    }

def create_charts(insights):
    """Generates all Plotly charts from the insights dictionary."""
    charts = {}
    
    df_equity = insights['equity_curve']
    fig_equity = px.line(df_equity, x='exit_time', y='equity_curve', title='Equity Curve', labels={'exit_time': 'Date', 'equity_curve': 'Equity ($)'})
    fig_equity.add_trace(go.Scatter(x=df_equity['exit_time'], y=df_equity['peak'], fill='tonexty', fillcolor='rgba(255,255,255,0.05)', line=dict(color='rgba(255,255,255,0.1)', width=1), name='Peak Equity'))
    fig_equity.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    charts['equity'] = fig_equity
    
    df_drawdown = insights['drawdown_curve']
    fig_drawdown = px.area(df_drawdown, x='exit_time', y='drawdown_usd', title='Drawdown ($)', labels={'exit_time': 'Date', 'drawdown_usd': 'Drawdown ($)'})
    fig_drawdown.update_traces(marker_color='#db4437', line_color='#db4437', fillcolor='rgba(219, 68, 55, 0.2)')
    fig_drawdown.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    charts['drawdown'] = fig_drawdown

    fig_pnl_dist = px.histogram(insights['pnl_distribution'], x='pnl', nbins=50, title='PnL Distribution per Trade', labels={'pnl': 'PnL ($)'}, color_discrete_sequence=['#4285f4'])
    fig_pnl_dist.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    charts['pnl_dist'] = fig_pnl_dist

    fig_side = px.bar(insights['side_performance'], title='Net PnL by Trade Side', labels={'index': 'Side', 'value': 'Net PnL ($)'})
    fig_side.update_traces(marker_color=['#0f9d58', '#db4437'])
    fig_side.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    charts['side'] = fig_side

    fig_day = px.bar(insights['day_of_week_performance'], title='Net PnL by Day of Week', labels={'index': 'Day', 'value': 'Net PnL ($)'})
    fig_day.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    charts['day'] = fig_day
    
    fig_hour = px.bar(insights['hour_of_day_performance'], title='Net PnL by Entry Hour (US/Eastern)', labels={'index': 'Hour', 'value': 'Net PnL ($)'})
    fig_hour.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    charts['hour'] = fig_hour
    
    return charts

@st.cache_data
def plot_trade_chart( _df_full, trade_data_dict, symbol ):
    """Generates the labeled mplfinance chart for a single trade."""
    
    trade = pd.Series(trade_data_dict)
    
    # Convert trade times (which are naive) back to UTC to match the main DF
    entry_time = pd.to_datetime(trade['entry_time']).tz_localize('UTC')
    exit_time = pd.to_datetime(trade['exit_time']).tz_localize('UTC')

    # --- **** THIS IS THE FIX (v12) **** ---
    pos_entry = _df_full.index.get_indexer([entry_time], method='nearest')[0]
    pos_exit = _df_full.index.get_indexer([exit_time], method='nearest')[0]

    left = max(0, pos_entry - 30)
    right = min(len(_df_full) - 1, pos_exit + 15)
    seg = _df_full.iloc[left:right + 1].copy()

    entry_idx_local = _df_full.index.get_indexer([entry_time], method='nearest')[0] - left
    exit_idx_local = _df_full.index.get_indexer([exit_time], method='nearest')[0] - left
    # --- **** END OF FIX **** ---
    
    hlines_to_plot = []
    hcolors = []
    hstyles = []
    
    # --- **** FIB LINES COMMENTED OUT **** ---
    # swing_low = trade.get('swing_low')
    # swing_high = trade.get('swing_high')
    # fib_levels_data = []
    
    # if swing_low and swing_high:
    #     swing_range = swing_high - swing_low
    #     fib_levels_data = [
    #         (swing_low, f"Swing Low (0.0)", '#E53935', '--'),
    #         (swing_low + swing_range * 0.5, "0.5", '#1E88E5', 'dotted'),
    #         (swing_low + swing_range * 0.618, "0.618 (OTE)", '#4CAF50', 'dotted'),
    #         (swing_low + swing_range * 0.786, "0.786 (OTE)", '#4CAF50', 'dotted'),
    #         (swing_high, f"Swing High (1.0)", '#E53935', '--')
    #     ]
    #     for level, _, color, style in fib_levels_data:
    #         hlines_to_plot.append(level)
    #         hcolors.append(color)
    #         hstyles.append(style)
    # --- **** END OF FIB LINES **** ---
    
    fig, axlist = mpf.plot(
        seg,
        type='candle',
        style='yahoo',
        figsize=(12, 6),
        returnfig=True,
        volume=False,
        show_nontrading=False,
        hlines=dict(hlines=hlines_to_plot, colors=hcolors, alpha=0.7, linestyle=hstyles)
    )
    ax = axlist[0]
    
    region_start = entry_idx_local
    region_end = exit_idx_local
    
    ax.fill_between([region_start, region_end], trade['entry'], trade['tp'], color='#0f9d58', alpha=0.1)
    ax.fill_between([region_start, region_end], trade['entry'], trade['sl'], color='#db4437', alpha=0.1)

    ax.scatter(entry_idx_local, trade['entry'], marker='^' if trade['side'] == 'BUY' else 'v', s=100, color='#4285f4',
               edgecolors='black', zorder=10)
    ax.scatter(exit_idx_local, trade['exit'], marker='x', s=80, color='orange', edgecolors='black', zorder=10)

    label_x_pos = len(seg) - 0.5
    
    # --- **** FIB LABELS COMMENTED OUT **** ---
    # for level, label, color, _ in fib_levels_data:
    #     ax.text(label_x_pos, level, f" {label}", 
    #             color=color, va='center', ha='left', fontsize=9, fontweight='bold')
    # --- **** END OF FIB LABELS **** ---
                
    ax.text(label_x_pos, trade['tp'], f" TP ({trade['tp']:.2f})", 
            color='#0f9d58', va='center', ha='left', fontsize=9, fontweight='bold')
    ax.text(label_x_pos, trade['sl'], f" SL ({trade['sl']:.2f})", 
            color='#db4437', va='center', ha='left', fontsize=9, fontweight='bold')

    tick_indices = np.linspace(0, len(seg) - 1, 6, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([seg.index[i].strftime('%m-%d\n%H:%M') for i in tick_indices])
    
    ax.set_xlim(-1, len(seg) + 3)
    ax.set_ylabel(f'Price ({symbol})')
    fig.tight_layout()
    
    return fig

# -----------------------------------------------------------------
# STREAMLIT APP UI
# -----------------------------------------------------------------

st.title("⚡ OTE Liquidity Sweep Strategy Dashboard")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Backtest Configuration")

# --- FIX: Only show 'Live MT5' if the library imported successfully ---
data_source_options = ["Offline File"]
if MT5_ENABLED:
    data_source_options.append("Live MT5")

data_source = st.sidebar.selectbox("Data Source", data_source_options, index=0) # Default to Offline
if data_source == "Live MT5" and not MT5_ENABLED:
    st.sidebar.error("Live MT5 mode is not available (library failed to import).")
    st.stop()


# --- ROBUST FOLDER/FILE CHECKING ---
offline_symbols = []
data_dir = './' # Look in current folder
app_ready = True

if not os.path.exists(data_dir):
    st.sidebar.error(f"Error: The app folder was not found.")
    app_ready = False
else:
    data_files = os.listdir(data_dir) # Look in current folder
    offline_symbols = list(set([f.split('_')[0] for f in data_files if f.endswith('.csv')]))
    offline_symbols.sort()

default_symbol_index = 0
if "XAUUSD" in offline_symbols:
    default_symbol_index = offline_symbols.index("XAUUSD")

# --- **** THIS IS THE FIX: Added months_to_test to sidebar **** ---
if data_source == "Offline File":
    if not offline_symbols:
        st.sidebar.error("No CSV files found in this folder.")
        st.info("Please add your exported data (e.g., 'XAUUSD_M5.csv') to this folder and refresh.")
        app_ready = False
    else:
        symbol = st.sidebar.selectbox("Select Symbol", offline_symbols, index=default_symbol_index)
        timeframe = "M5" # Hardcoded
        months_to_test = 0 # Not used for offline
else:
    symbol = st.sidebar.text_input("Symbol", "XAUUSD")
    timeframe = st.sidebar.selectbox("Timeframe", ["M5", "M15", "H1"])
    months_to_test = st.sidebar.slider("Months to Backtest", 1, 24, 9)
# --- **** END OF FIX **** ---


# Get parameters from Python's BASE config
base_params = {
    'lookback_swing': 20,
    'ote_low': 0.618,
    'ote_high': 0.79,
    'rr': 2.0,
    'atr_period': 14,
    'atr_sl_multiplier': 1.5,
    'min_swing_pips': 15.0,
    'risk_per_trade': 0.01,
    'initial_equity': 10000.0,
    'ema_trend_period': 200
}

base_params['rr'] = st.sidebar.number_input("Risk:Reward Ratio", min_value=0.5, max_value=10.0, value=base_params['rr'], step=0.1)

# --- START BACKTEST BUTTON ---
if app_ready:
    if st.sidebar.button("🚀 Start Backtest", type="primary"):
        
        with st.spinner(f"Running backtest for {symbol} on {timeframe}..."):
            
            # 1. Load Data
            df, test_start_date = (None, None)
            if data_source == "Live MT5":
                df, test_start_date = fetch_mt5_data(symbol, timeframe, months_to_test)
            else: # Offline File
                df, test_start_date = load_offline_data(symbol, timeframe)

            if df is None:
                st.error("Failed to load data. Backtest aborted.")
            else:
                st.success(f"Loaded {len(df)} bars of data. Starting analysis...")
                
                # 2. Get Symbol Info
                symbol_info = get_symbol_info(symbol)
                if symbol_info is None:
                    st.warning(f"Could not get symbol info for {symbol} (MT5 not connected?). Using fallback defaults for lot sizing.")
                    
                    # --- **** CRYPTO FIX (v4): Correct fallback logic for ETHUSD **** ---
                    if symbol == "XAUUSD":
                        symbol_info = {'pip_size': 0.01, 'point': 0.01, 'lot_value_per_point': 100.0, 'digits': 2}
                    elif symbol == "ETHUSD": # Crypto
                         symbol_info = {'pip_size': 0.01, 'point': 0.0001, 'lot_value_per_point': 10.0, 'digits': 2}
                    elif symbol == "BTCUSD": # Crypto
                         symbol_info = {'pip_size': 0.01, 'point': 0.01, 'lot_value_per_point': 1.0, 'digits': 2}
                    elif "USD" in symbol: # Forex
                        symbol_info = {'pip_size': 0.0001, 'point': 0.00001, 'lot_value_per_point': 100000.0, 'digits': 5}
                    else: # Other fallback
                         symbol_info = {'pip_size': 0.01, 'point': 0.01, 'lot_value_per_point': 1.0, 'digits': 2}
                
                # 3. Run Backtest
                trades_df = run_backtest(
                    df, 
                    base_params, 
                    symbol_info, 
                    test_start_date, 
                    require_fvg=False # Using your most profitable logic
                )
                
                # 4. Calculate Insights
                if trades_df.empty:
                    st.warning("Backtest ran successfully, but no trades were found with these parameters.")
                else:
                    st.success(f"Backtest complete! Found {len(trades_df)} trades.")
                    
                    # Store full df in session state for charting
                    st.session_state['full_df'] = df
                    st.session_state['trades_df'] = trades_df
                    
                    insights = calculate_insights(trades_df, base_params['initial_equity'])
                    charts = create_charts(insights)
                    
                    # 5. Display KPIs
                    st.header("Key Performance Indicators (KPIs)")
                    cols = st.columns(6)
                    pnl_color = 'normal' if insights['net_pnl'] > 0 else 'inverse'
                    pf_color = 'normal' if insights['profit_factor'] > 1 else 'inverse'
                    exp_color = 'normal' if insights['expectancy'] > 0 else 'inverse'
                    
                    cols[0].metric("Net Profit / Loss ($)", f"{insights['net_pnl']:,.2f}", delta_color=pnl_color)
                    cols[1].metric("Win Rate (%)", f"{insights['win_rate']:.2f}")
                    cols[2].metric("Total Trades", f"{insights['total_trades']}")
                    cols[3].metric("Profit Factor", f"{insights['profit_factor']:.2f}", delta_color=pf_color)
                    cols[4].metric("Expectancy ($ / Trade)", f"{insights['expectancy']:.2f}", delta_color=exp_color)
                    cols[5].metric("Max Drawdown ($)", f"{insights['max_drawdown_usd']:,.2f}", delta_color="inverse")
                    
                    # 6. Display Equity & Drawdown
                    st.header("Equity & Drawdown")
                    st.plotly_chart(charts['equity'], use_container_width=True)
                    st.plotly_chart(charts['drawdown'], use_container_width=True)
                    
                    # 7. Display Analytics
                    st.header("Trade Analytics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(charts['pnl_dist'], use_container_width=True)
                        st.plotly_chart(charts['day'], use_container_width=True)
                    with col2:
                        st.plotly_chart(charts['side'], use_container_width=True)
                        st.plotly_chart(charts['hour'], use_container_width=True)

                    # 8. Display Trade Log
                    st.header("Full Trade Log")
                    
                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv().encode('utf-8')

                    csv_data = convert_df_to_csv(trades_df)
                    st.download_button(
                        label="Download Trades as CSV",
                        data=csv_data,
                        file_name=f"{symbol}_{timeframe}_trade_log.csv",
                        mime="text/csv",
                    )
                    
                    display_cols = ['entry_time', 'exit_time', 'side', 'lot', 'entry', 'sl', 'tp', 'exit', 'result', 'pnl', 'reason']
                    st.dataframe(trades_df[display_cols])

                    st.subheader("Individual Trade Charts")
                    st.warning("Click the expander to generate and view the chart for a specific trade.")
                    
                    # Create a container for each chart
                    for trade_id, trade in trades_df.iterrows():
                        expander = st.expander(f"Trade #{trade_id}: {trade['side']} | PnL: ${trade['pnl']:.2f} | Reason: {trade['reason']}")
                        with expander:
                            # We generate the chart *only* when the user clicks
                            try:
                                # Convert trade Series to dict for caching
                                fig = plot_trade_chart(df, trade.to_dict(), symbol)
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Could not plot chart for trade #{trade_id}: {e}")

    else:
        st.info("Configuration set. Click 'Start Backtest' in the sidebar to run your analysis.")
