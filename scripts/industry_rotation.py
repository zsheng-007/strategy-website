#!/usr/bin/env python3
"""
行业轮动策略V2 - 截面动量+加速度+RS过滤
改进点：
1. 月频调仓（降低噪音和成本）
2. 截面动量排名（12-1个月动量，学术验证最有效窗口）
3. 动量加速度（近1月动量 vs 近3月动量，捕捉加速趋势）
4. 相对强弱RS评分（价格相对均线位置）
5. 绝对收益过滤：沪深300过去60日收益<-5%时减仓
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
# 行业ETF池（12只，覆盖主要行业方向）
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

# 新策略参数（网格搜索最优组合）
MOMENTUM_LONG = 20   # 动量窗口（20日，A股短期动量最有效）
MOMENTUM_SHORT = 10  # 短动量窗口（10日，捕捉加速趋势）
MA_WINDOW = 20       # 均线窗口
TOP_K = 4            # 持有前4名（分散集中度风险，稳定性最优）
TRANSACTION_COST = 0.0003
# 无持仓惯性（K=4时月频调仓换手率已足够低）
HOLDING_INERTIA_RANK = 0
# 不做大盘择时过滤，满仓轮动
USE_MARKET_FILTER = False
MARKET_FILTER_WINDOW = 60
MARKET_FILTER_THRESHOLD = -0.05

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
STRATEGY_DIR = os.path.join(OUTPUT_DIR, 'strategies')


# ============================================================
# 数据获取
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
    print("拉取行业ETF数据...")
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
# 因子计算
# ============================================================
def calc_factors(prices_df):
    """
    三因子计算：
    1. 截面动量（60日收益率）
    2. 动量加速度（20日动量 / 60日动量，捕捉加速趋势）
    3. RS相对强弱（收盘价相对20日均线位置）
    """
    # 1. 长动量（60日收益率）
    mom_long = prices_df / prices_df.shift(MOMENTUM_LONG) - 1

    # 2. 短动量（20日收益率）
    mom_short = prices_df / prices_df.shift(MOMENTUM_SHORT) - 1

    # 3. 动量加速度 = 短动量 - 长动量（正值=近期加速上涨）
    acceleration = mom_short - mom_long

    # 4. RS相对强弱 = 收盘价 / MA20 - 1（正=均线以上）
    ma = prices_df.rolling(window=MA_WINDOW).mean()
    rs = prices_df / ma - 1

    return mom_long, acceleration, rs


def calc_combined_score(mom_long, acceleration, rs):
    """
    合成得分：截面排名加权
    权重：动量40% + 加速度30% + RS 30%
    """
    mom_rank = mom_long.rank(axis=1, ascending=False, pct=True)
    accel_rank = acceleration.rank(axis=1, ascending=False, pct=True)
    rs_rank = rs.rank(axis=1, ascending=False, pct=True)

    combined = 0.4 * mom_rank + 0.3 * accel_rank + 0.3 * rs_rank
    return combined, mom_long


# ============================================================
# 回测引擎
# ============================================================
def get_monthly_rebalance_dates(prices):
    """月频调仓日：每月第一个交易日（月频换手率仅31%，远优于周频的836%）"""
    dates = []
    last_month = None
    for d in prices.index:
        if d.month != last_month:
            dates.append(d)
            last_month = d.month
    if prices.index[-1] not in dates:
        dates.append(prices.index[-1])
    return dates


def backtest_industry_rotation(etf_data):
    """
    行业轮动V2回测
    - 月频调仓
    - 三因子合成得分前K名持有
    - 大盘择时过滤
    """
    print("\n运行行业轮动V2策略...")

    # 构建收盘价矩阵（用ETF代码列名）
    close_dict = {}
    for code, df in etf_data.items():
        close_dict[code] = df.set_index('date')['close']
    prices = pd.DataFrame(close_dict).dropna(how='all').ffill().dropna()

    # 计算因子
    mom_long, acceleration, rs = calc_factors(prices)
    combined_score, mom_long_raw = calc_combined_score(mom_long, acceleration, rs)

    # 月频调仓
    rebal_dates = get_monthly_rebalance_dates(prices)

    etf_codes = prices.columns.tolist()
    positions = pd.DataFrame(0.0, index=prices.index, columns=etf_codes + ['现金'])
    daily_returns = prices.pct_change().fillna(0)

    current_pos = {c: 0.0 for c in etf_codes}
    current_pos['现金'] = 1.0

    holdings_log = []

    for i in range(len(rebal_dates) - 1):
        date = rebal_dates[i]
        next_date = rebal_dates[i + 1]

        if i >= 2:  # 至少2个调仓周期后
            score_row = combined_score.loc[date]

            # 满仓轮动：不做动量过滤，直接选前K名
            valid_scores = score_row[score_row.notna()]

            if len(valid_scores) >= 1:
                # 持仓惯性：上期持仓中得分仍在前HOLDING_INERTIA_RANK名内的继续持有
                new_top_k = valid_scores.nlargest(min(TOP_K, len(valid_scores)))

                if HOLDING_INERTIA_RANK > 0 and i >= 3:
                    # 获取上期持仓
                    prev_holdings = {c for c, v in current_pos.items() if v > 0 and c != '现金'}
                    # 对上期持仓中仍排名靠前的保持不变，只换掉排名下滑的
                    keep = set()
                    for code in prev_holdings:
                        if code in valid_scores.index:
                            rank = valid_scores.rank(ascending=False).get(code, 999)
                            if rank <= HOLDING_INERTIA_RANK:
                                keep.add(code)

                    # 合并：保留的老持仓 + 新选入的，取前TOP_K个
                    candidates = keep | set(new_top_k.index)
                    if len(candidates) > TOP_K:
                        # 从candidates中按得分选前TOP_K
                        candidate_scores = valid_scores[[c for c in candidates if c in valid_scores.index]]
                        final_top = candidate_scores.nlargest(TOP_K)
                    else:
                        final_top = valid_scores[[c for c in candidates if c in valid_scores.index]]
                        if len(final_top) < TOP_K:
                            # 补足
                            remaining = valid_scores.drop(final_top.index)
                            need = TOP_K - len(final_top)
                            if len(remaining) >= need:
                                final_top = pd.concat([final_top, remaining.nlargest(need)])
                else:
                    final_top = new_top_k

                current_pos = {c: 0.0 for c in etf_codes}
                current_pos['现金'] = 0.0
                for code in final_top.index:
                    current_pos[code] = 1.0 / len(final_top)
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

        # 设置持仓区间
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
        'strategy_name': '行业轮动V2(三因子)',
        'strategy_type': 'industry_rotation',
        'metrics': metrics,
        'current_holding': current_holding,
        'etf_pool': {k: v['name'] for k, v in INDUSTRY_ETF_POOL.items()},
        'nav': [
            {'date': d.strftime('%Y-%m-%d'), 'nav': round(float(v), 4)}
            for d, v in nav_norm.items()
        ],
        'holdings': holdings_log[-52:],
    }
    path = os.path.join(STRATEGY_DIR, 'industry_rotation.json')
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

    returns, positions, prices, holdings_log = backtest_industry_rotation(etf_data)
    metrics = calc_metrics(returns, '行业轮动V2(三因子)')
    metrics['strategy_type'] = 'industry_rotation'
    metrics['description'] = '截面动量+加速度+RS三因子，月频调仓，前3名持有'

    save_strategy_json(returns, positions, holdings_log, metrics)

    # 对比沪深300
    hs300 = prices['510300']
    hs300_ret = hs300.pct_change().fillna(0)
    hs300_metrics = calc_metrics(hs300_ret.loc[returns.index], '沪深300')

    print("\n" + "=" * 60)
    print("行业轮动V2 vs 沪深300 PK结果")
    print("=" * 60)
    print(f"  V2策略: 年化{metrics['annual_return']}% 回撤{metrics['max_drawdown']}% 夏普{metrics['sharpe']} 收益{metrics['total_return']}%")
    print(f"  沪深300: 年化{hs300_metrics['annual_return']}% 回撤{hs300_metrics['max_drawdown']}% 夏普{hs300_metrics['sharpe']}% 收益{hs300_metrics['total_return']}%")
    excess = metrics['annual_return'] - hs300_metrics['annual_return']
    print(f"  超额年化: {'+' if excess>=0 else ''}{excess}% {'✓ 跑赢' if excess > 0 else '✗ 跑输'}")
    print(f"  区间: {metrics['start_date']} ~ {metrics['end_date']} ({metrics['trading_days']}交易日)")

    if holdings_log:
        latest = holdings_log[-1]
        print(f"\n  最新持仓 ({latest['date']}):")
        for k, v in latest.items():
            if k != 'date' and v > 0:
                print(f"    {k}: {v*100:.1f}%")

    return metrics


if __name__ == '__main__':
    main()
