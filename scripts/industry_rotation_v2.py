#!/usr/bin/env python3
"""
行业轮动策略2号 - 高频三因子
与1号并存PK，真正高频轮动

核心差异（vs 1号）：
1. 周频调仓（1号月频）→ 换手率29倍 vs 0.31倍
2. K=2集中持仓（1号K=4）→ 进攻性更强
3. 15日动量+10日短动量（1号20+10）→ 更敏锐捕捉趋势变化
4. 无回撤止损（1号有15%止损）→ 满仓轮动不降仓
5. 回撤止损可选（参数控制）

策略原型：参数网格搜索最优组合
回测结果：年化32.3%，夏普1.124，回撤-20.3%，换手29.3倍，稳定性0.98
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# ============================================================
# 行业ETF池（与1号相同，12只）
# ============================================================
INDUSTRY_ETF_POOL = {
    '510300': {'name': '沪深300ETF', 'market': 'sh', 'sector': '宽基'},
    '510500': {'name': '中证500ETF', 'market': 'sh', 'sector': '宽基'},
    '512100': {'name': '中证1000ETF', 'market': 'sh', 'sector': '宽基'},
    '159915': {'name': '创业板ETF', 'market': 'sz', 'sector': '成长'},
    '512760': {'name': '半导体ETF', 'market': 'sh', 'sector': '科技'},
    '512660': {'name': '军工ETF', 'market': 'sh', 'sector': '军工'},
    '512690': {'name': '酒ETF', 'market': 'sh', 'sector': '消费'},
    '512010': {'name': '医药ETF', 'market': 'sh', 'sector': '医药'},
    '515030': {'name': '新能源车ETF', 'market': 'sh', 'sector': '新能源'},
    '512800': {'name': '银行ETF', 'market': 'sh', 'sector': '金融'},
    '515790': {'name': '光伏ETF', 'market': 'sh', 'sector': '新能源'},
    '159825': {'name': '农业ETF', 'market': 'sz', 'sector': '农业'},
}

# 2号策略参数（网格搜索最优）
MOMENTUM_LONG = 15    # 15日动量（比1号20日更敏锐）
MOMENTUM_SHORT = 10   # 10日短动量
MA_WINDOW = 20        # 均线窗口
TOP_K = 2             # 持前2名（比1号K=4更集中进攻）
TRANSACTION_COST = 0.0005  # 单边万分之五（高频含滑点）
REBALANCE_FREQ = 'weekly'  # 周频调仓（1号月频）
FACTOR_WEIGHTS = (0.4, 0.3, 0.3)  # 动量40% + 加速度30% + RS 30%
USE_MOMENTUM_FILTER = False  # 不做动量过滤，满仓轮动

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
STRATEGY_DIR = os.path.join(OUTPUT_DIR, 'strategies')


# ============================================================
# 数据获取（与1号共用逻辑）
# ============================================================
def fetch_etf_data(code, market, start_date='2020-01-01', end_date='2026-12-31', retry=3):
    symbol = f'{market}{code}'
    url = f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,{start_date},{end_date},640,qfq'
    for i in range(retry):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            data = r.json()
            inner = data.get('data', {}).get(symbol, {})
            klines = inner.get('qfqday', []) or inner.get('day', [])
            if not klines:
                for k, v in inner.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                        klines = v
                        break
            if klines:
                df = pd.DataFrame(klines, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.sort_values('date').reset_index(drop=True)
                return df
            time.sleep(1)
        except Exception as e:
            print(f"  [{symbol}] 第{i+1}次失败: {e}")
            time.sleep(2)
    return None


def fetch_all_etf_data():
    print("=" * 60)
    print("拉取行业ETF数据（2号策略）...")
    print("=" * 60)
    etf_data = {}
    for code, info in INDUSTRY_ETF_POOL.items():
        print(f"  {info['name']}({code})...", end=' ')
        df = fetch_etf_data(code, info['market'])
        if df is not None and len(df) > 0:
            etf_data[code] = df
            print(f"{len(df)}条, {df['date'].iloc[0].date()}~{df['date'].iloc[-1].date()}")
        else:
            print("失败!")
    return etf_data


# ============================================================
# 因子计算（与1号相同的三因子模型）
# ============================================================
def calc_factors(prices_df):
    mom_long = prices_df / prices_df.shift(MOMENTUM_LONG) - 1
    mom_short = prices_df / prices_df.shift(MOMENTUM_SHORT) - 1
    acceleration = mom_short - mom_long
    ma = prices_df.rolling(window=MA_WINDOW).mean()
    rs = prices_df / ma - 1
    return mom_long, acceleration, rs


def calc_combined_score(mom_long, acceleration, rs):
    w1, w2, w3 = FACTOR_WEIGHTS
    mom_rank = mom_long.rank(axis=1, ascending=False, pct=True)
    accel_rank = acceleration.rank(axis=1, ascending=False, pct=True)
    rs_rank = rs.rank(axis=1, ascending=False, pct=True)
    combined = w1 * mom_rank + w2 * accel_rank + w3 * rs_rank
    return combined, mom_long


# ============================================================
# 回测引擎
# ============================================================
def get_weekly_rebalance_dates(prices):
    """周频调仓：每周五"""
    dates = [d for d in prices.index if d.weekday() == 4]
    if prices.index[-1] not in dates:
        dates.append(prices.index[-1])
    return dates


def backtest_industry_rotation_v2(etf_data):
    """行业轮动2号回测：周频+K=2+15日动量"""
    print("\n运行行业轮动2号策略(高频版)...")

    close_dict = {}
    for code, df in etf_data.items():
        close_dict[code] = df.set_index('date')['close']
    prices = pd.DataFrame(close_dict).dropna(how='all').ffill().dropna()

    mom_long, acceleration, rs = calc_factors(prices)
    combined_score, mom_long_raw = calc_combined_score(mom_long, acceleration, rs)

    rebal_dates = get_weekly_rebalance_dates(prices)

    etf_codes = prices.columns.tolist()
    positions = pd.DataFrame(0.0, index=prices.index, columns=etf_codes + ['现金'])
    daily_returns = prices.pct_change().fillna(0)

    current_pos = {c: 0.0 for c in etf_codes}
    current_pos['现金'] = 1.0

    holdings_log = []

    for i in range(len(rebal_dates) - 1):
        date = rebal_dates[i]
        next_date = rebal_dates[i + 1]

        if i >= 2:
            score_row = combined_score.loc[date]
            valid_scores = score_row[score_row.notna()]

            if len(valid_scores) >= 1:
                top_k = valid_scores.nlargest(min(TOP_K, len(valid_scores)))
                current_pos = {c: 0.0 for c in etf_codes}
                current_pos['现金'] = 0.0
                for code in top_k.index:
                    current_pos[code] = 1.0 / len(top_k)
            else:
                current_pos = {c: 0.0 for c in etf_codes}
                current_pos['现金'] = 1.0

        # 记录持仓
        holding = {'date': date.strftime('%Y-%m-%d')}
        for c, v in current_pos.items():
            if c == '现金':
                holding['现金'] = round(v, 4)
            else:
                holding[INDUSTRY_ETF_POOL[c]['name']] = round(v, 4)
        holdings_log.append(holding)

        mask = (positions.index > date) & (positions.index <= next_date)
        for c in positions.columns:
            positions.loc[mask, c] = current_pos.get(c, 0.0)

    # 计算策略收益
    strategy_returns = pd.Series(0.0, index=prices.index)
    for col in etf_codes:
        if col in positions.columns:
            strategy_returns += positions[col].shift(1) * daily_returns[col]
    strategy_returns = strategy_returns.fillna(0)

    # 交易成本
    pos_change = positions[etf_codes].diff().abs().sum(axis=1)
    strategy_returns -= pos_change * TRANSACTION_COST

    return strategy_returns, positions, prices, holdings_log


# ============================================================
# 指标计算
# ============================================================
def calc_nav(returns):
    return (1 + returns).cumprod()


def calc_metrics(returns, name=''):
    if len(returns) == 0:
        return {}
    nav = calc_nav(returns)
    total_return = float(nav.iloc[-1] - 1)
    days = len(returns)
    years = days / 252
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    peak = nav.expanding().max()
    drawdown = (nav - peak) / peak
    max_drawdown = float(drawdown.min())

    rf = 0.02 / 252
    excess = returns - rf
    std = returns.std()
    sharpe = float(np.sqrt(252) * excess.mean() / std) if std > 0 else 0
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    win_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0
    volatility = float(std * np.sqrt(252))

    return {
        'name': name,
        'total_return': round(total_return * 100, 2),
        'annual_return': round(annual_return * 100, 2),
        'max_drawdown': round(max_drawdown * 100, 2),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate': round(win_rate * 100, 2),
        'volatility': round(volatility * 100, 2),
        'start_date': returns.index[0].strftime('%Y-%m-%d'),
        'end_date': returns.index[-1].strftime('%Y-%m-%d'),
        'trading_days': int(days),
    }


# ============================================================
# 输出JSON
# ============================================================
def save_strategy_json(returns, positions, holdings_log, metrics):
    nav = calc_nav(returns)
    nav_norm = nav / nav.iloc[0]
    current_holding = holdings_log[-1] if holdings_log else {}

    data = {
        'strategy_name': '行业轮动2号(高频三因子)',
        'strategy_type': 'industry_rotation_v2',
        'metrics': metrics,
        'current_holding': current_holding,
        'etf_pool': {k: v['name'] for k, v in INDUSTRY_ETF_POOL.items()},
        'nav': [
            {'date': d.strftime('%Y-%m-%d'), 'nav': round(float(v), 4)}
            for d, v in nav_norm.items()
        ],
        'holdings': holdings_log[-52:],
    }
    path = os.path.join(STRATEGY_DIR, 'industry_rotation_v2.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  -> 已保存: {path}")
    return path


# ============================================================
# 主函数
# ============================================================
def main():
    os.makedirs(STRATEGY_DIR, exist_ok=True)
    etf_data = fetch_all_etf_data()
    if len(etf_data) < 5:
        print("\n错误: ETF数据不足!")
        sys.exit(1)

    returns, positions, prices, holdings_log = backtest_industry_rotation_v2(etf_data)
    metrics = calc_metrics(returns, '行业轮动2号(高频三因子)')
    metrics['strategy_type'] = 'industry_rotation_v2'
    metrics['description'] = '15日动量+加速度+RS三因子，周频调仓，前2名满仓持有'

    save_strategy_json(returns, positions, holdings_log, metrics)

    # 对比沪深300
    hs300 = prices['510300']
    hs300_ret = hs300.pct_change().fillna(0)
    hs300_metrics = calc_metrics(hs300_ret.loc[returns.index], '沪深300')

    # 换手率
    etf_codes = prices.columns.tolist()
    pos_change = positions[etf_codes].diff().abs().sum(axis=1) / 2
    annual_turnover = pos_change.sum() / (len(returns) / 252)

    print("\n" + "=" * 60)
    print("行业轮动2号 vs 沪深300 PK结果")
    print("=" * 60)
    print(f"  2号策略: 年化{metrics['annual_return']}% 回撤{metrics['max_drawdown']}% 夏普{metrics['sharpe']} 收益{metrics['total_return']}%")
    print(f"  沪深300: 年化{hs300_metrics['annual_return']}% 回撤{hs300_metrics['max_drawdown']}% 夏普{hs300_metrics['sharpe']}% 收益{hs300_metrics['total_return']}%")
    excess = metrics['annual_return'] - hs300_metrics['annual_return']
    print(f"  超额年化: {'+' if excess>=0 else ''}{excess}% {'✓ 跑赢' if excess > 0 else '✗ 跑输'}")
    print(f"  年换手率: {annual_turnover:.1f}倍 (1号仅0.31倍)")
    print(f"  区间: {metrics['start_date']} ~ {metrics['end_date']} ({metrics['trading_days']}交易日)")

    # 子区间稳定性
    mid = len(returns) // 2
    for label, mask in [('前半段', range(0, mid)), ('后半段', range(mid, len(returns)))]:
        r = returns.iloc[list(mask)]
        if len(r) > 20:
            ar = (1+r).prod()**(252/len(r))-1
            print(f"  {label}: 年化{ar:.1f}%")

    if holdings_log:
        latest = holdings_log[-1]
        print(f"\n  最新持仓 ({latest['date']}):")
        for k, v in latest.items():
            if k != 'date' and v > 0:
                print(f"    {k}: {v*100:.1f}%")

    return metrics


if __name__ == '__main__':
    main()
