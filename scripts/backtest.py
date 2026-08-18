#!/usr/bin/env python3
"""
志胜投资策略 - 数据获取与回测引擎
标的: 价值ETF(512040) + 成长ETF(159259)
基准: 中证A500(000922) + 中证500(000002,作为偏股混合指数885001的替代)
数据源: 腾讯行情API
调仓频率: 周频（每周五）
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

# ============================================================
# 配置
# ============================================================
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

ETF_CONFIG = {
    '512040': {
        'name': '价值ETF',
        'fullname': '价值ETF富国',
        'market': 'sh',
        'track_index': '国证价值100(980013)',
        'fund_date': '2018-11-15',
    },
    '159259': {
        'name': '成长ETF',
        'fullname': '成长ETF易方达',
        'market': 'sz',
        'track_index': '国证成长100(980080)',
        'fund_date': '2025-08-20',
    },
}

BENCHMARK_CONFIG = {
    '000922': {'name': '中证A500', 'market': 'sh', 'file': 'a500'},
    'us.INX': {'name': '标普500', 'market': 'us', 'file': 'sp500'},
}

# 偏股混合基金指数885001(Wind独家)无法从免费源获取，改用标普500作为海外基准

REBALANCE_WEEKDAY = 4  # 周五
MOMENTUM_LOOKBACK = 4  # 动量回看周数
TRANSACTION_COST = 0.001  # 单边交易成本0.1%

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
STRATEGY_DIR = os.path.join(OUTPUT_DIR, 'strategies')
BENCHMARK_DIR = os.path.join(OUTPUT_DIR, 'benchmarks')


# ============================================================
# 数据获取 - 腾讯行情API
# ============================================================
def fetch_kline_tencent(code, market, start_date='2018-01-01', end_date='2026-12-31', adjust='qfq', retry=3):
    """从腾讯API获取日K线数据
    A股: market=sh/sz, code=数字代码 → symbol=sh512040
    美股: market=us, code=us.INX → symbol=us.INX
    """
    if market == 'us':
        symbol = code  # 美股code本身就是完整symbol（如us.INX）
    else:
        symbol = f'{market}{code}'
    # 腾讯API日频最多返回640条
    url = f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,{start_date},{end_date},640,{adjust}'

    for i in range(retry):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            data = r.json()
            inner = data.get('data', {}).get(symbol, {})

            # 数据在 qfqday(前复权) 或 day(不复权) 键下
            data_key = f'{adjust}day' if adjust else 'day'
            klines = inner.get(data_key, []) or inner.get('day', [])

            if not klines:
                # 尝试不带前缀
                for k, v in inner.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                        klines = v
                        break

            if klines:
                # 格式: [date, open, close, high, low, volume]
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


def fetch_all_data():
    """拉取全部ETF和指数数据"""
    print("=" * 60)
    print("开始拉取行情数据（腾讯API）...")
    print("=" * 60)

    etf_data = {}
    for code, info in ETF_CONFIG.items():
        print(f"\n拉取ETF: {info['fullname']}({code})")
        df = fetch_kline_tencent(code, info['market'])
        if df is not None and len(df) > 0:
            etf_data[code] = df
            print(f"  成功: {len(df)}条, {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
        else:
            print(f"  失败!")

    benchmark_data = {}
    for code, info in BENCHMARK_CONFIG.items():
        print(f"\n拉取指数: {info['name']}({code})")
        df = fetch_kline_tencent(code, info['market'], adjust='')
        if df is not None and len(df) > 0:
            benchmark_data[code] = df
            print(f"  成功: {len(df)}条, {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
        else:
            print(f"  失败!")

    return etf_data, benchmark_data


# ============================================================
# 数据预处理
# ============================================================
def prepare_prices(etf_data):
    """合并ETF收盘价，对齐日期"""
    price_dict = {}
    for code, df in etf_data.items():
        price_dict[ETF_CONFIG[code]['name']] = df.set_index('date')['close']

    prices = pd.DataFrame(price_dict).dropna(how='all')
    # 前向填充（停牌等情况）
    prices = prices.ffill().dropna()
    return prices


def prepare_benchmarks(benchmark_data):
    """处理基准指数数据"""
    bench_dict = {}
    for code, df in benchmark_data.items():
        name = BENCHMARK_CONFIG[code]['name']
        bench_dict[name] = df.set_index('date')['close']

    bench = pd.DataFrame(bench_dict).dropna(how='all').ffill().dropna()
    return bench


# ============================================================
# 回测核心
# ============================================================
def calc_nav(returns):
    """从日收益率计算净值"""
    return (1 + returns).cumprod()


def calc_metrics(returns, name=''):
    """计算回测指标"""
    if len(returns) == 0:
        return {}
    nav = calc_nav(returns)
    total_return = float(nav.iloc[-1] - 1)

    days = len(returns)
    years = days / 252
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 最大回撤
    peak = nav.expanding().max()
    drawdown = (nav - peak) / peak
    max_drawdown = float(drawdown.min())

    # 夏普比率（无风险利率2%）
    rf = 0.02 / 252
    excess = returns - rf
    std = returns.std()
    sharpe = float(np.sqrt(252) * excess.mean() / std) if std > 0 else 0

    # Calmar比率
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # 胜率
    win_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0

    # 波动率
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


def get_rebalance_dates(prices, weekday=REBALANCE_WEEKDAY):
    """获取调仓日（每周指定weekday）"""
    dates = []
    for d in prices.index:
        if d.weekday() == weekday:
            dates.append(d)
    # 包含最后一个交易日
    if prices.index[-1] not in dates:
        dates.append(prices.index[-1])
    return dates


# ============================================================
# 策略实现
# ============================================================
def strategy_momentum_rotation(prices):
    """
    策略A - 动量轮动
    每周五比较过去N周动量，满仓强势ETF
    """
    print("  运行策略A: 动量轮动...")
    rebal_dates = get_rebalance_dates(prices)
    etf_cols = prices.columns.tolist()

    positions = pd.DataFrame(0.5, index=prices.index, columns=etf_cols)
    daily_returns = prices.pct_change().fillna(0)

    current_pos = {c: 0.5 for c in etf_cols}

    for i in range(len(rebal_dates) - 1):
        date = rebal_dates[i]
        next_date = rebal_dates[i + 1]

        if i >= MOMENTUM_LOOKBACK:
            # 计算过去N周动量
            lookback_date = rebal_dates[i - MOMENTUM_LOOKBACK]
            momentum = prices.loc[date] / prices.loc[lookback_date] - 1

            # 满仓强势ETF
            winner = momentum.idxmax()
            current_pos = {c: 0.0 for c in etf_cols}
            current_pos[winner] = 1.0

        # 设置持仓区间
        mask = (positions.index > date) & (positions.index <= next_date)
        for c in etf_cols:
            positions.loc[mask, c] = current_pos[c]

    strategy_returns = (positions.shift(1) * daily_returns).sum(axis=1).fillna(0)
    # 交易成本
    pos_change = positions.diff().abs().sum(axis=1)
    strategy_returns -= pos_change * TRANSACTION_COST

    return strategy_returns, positions


def strategy_equal_weight(prices):
    """
    策略B - 等权再平衡
    50/50每周再平衡
    """
    print("  运行策略B: 等权再平衡...")
    etf_cols = prices.columns.tolist()
    positions = pd.DataFrame(0.5, index=prices.index, columns=etf_cols)
    daily_returns = prices.pct_change().fillna(0)

    strategy_returns = (positions.shift(1) * daily_returns).sum(axis=1).fillna(0)

    # 每次再平衡的成本（偏离后回归50/50）
    rebal_dates = get_rebalance_dates(prices)
    for i in range(len(rebal_dates) - 1):
        date = rebal_dates[i]
        # 计算偏离程度
        actual_pos = positions.loc[date]
        # 偏离越大成本越高
        deviation = (actual_pos - 0.5).abs().sum()
        strategy_returns.loc[date] -= deviation * TRANSACTION_COST

    return strategy_returns, positions


def strategy_relative_strength(prices):
    """
    策略C - 相对强弱动态配比
    基于动量强弱动态调整权重，限制在20%-80%之间
    """
    print("  运行策略C: 相对强弱动态配比...")
    rebal_dates = get_rebalance_dates(prices)
    etf_cols = prices.columns.tolist()

    positions = pd.DataFrame(0.5, index=prices.index, columns=etf_cols)
    daily_returns = prices.pct_change().fillna(0)

    current_pos = {c: 0.5 for c in etf_cols}

    for i in range(len(rebal_dates) - 1):
        date = rebal_dates[i]
        next_date = rebal_dates[i + 1]

        if i >= MOMENTUM_LOOKBACK:
            lookback_date = rebal_dates[i - MOMENTUM_LOOKBACK]
            momentum = prices.loc[date] / prices.loc[lookback_date] - 1

            mom_values = np.array([momentum[c] for c in etf_cols])

            # 正动量按比例分配，限制在20%-80%
            positive_mom = np.maximum(mom_values, 0)
            total_pos = positive_mom.sum()

            if total_pos > 0:
                weights = positive_mom / total_pos
                # 限制极端权重
                weights = np.clip(weights, 0.2, 0.8)
                weights = weights / weights.sum()
            else:
                # 全部为负时等权
                weights = np.array([0.5] * len(etf_cols))

            current_pos = {c: float(weights[j]) for j, c in enumerate(etf_cols)}

        mask = (positions.index > date) & (positions.index <= next_date)
        for c in etf_cols:
            positions.loc[mask, c] = current_pos[c]

    strategy_returns = (positions.shift(1) * daily_returns).sum(axis=1).fillna(0)
    pos_change = positions.diff().abs().sum(axis=1)
    strategy_returns -= pos_change * TRANSACTION_COST

    return strategy_returns, positions


# ============================================================
# 输出JSON
# ============================================================
def strategy_to_json(nav, name, stype, metrics, positions=None):
    """策略净值转JSON"""
    nav_norm = nav / nav.iloc[0]

    data = {
        'strategy_name': name,
        'strategy_type': stype,
        'metrics': metrics,
        'nav': [
            {
                'date': d.strftime('%Y-%m-%d'),
                'nav': round(float(v), 4),
            }
            for d, v in nav_norm.items()
        ],
    }

    # 持仓记录（最近52周）
    if positions is not None:
        holdings = []
        rebal_dates = [d for d in positions.index if d.weekday() == REBALANCE_WEEKDAY]
        for d in rebal_dates[-52:]:
            h = {'date': d.strftime('%Y-%m-%d')}
            for c in positions.columns:
                h[c] = round(float(positions.loc[d, c]), 4)
            holdings.append(h)
        data['holdings'] = holdings

    return data


def benchmark_to_json(series, name, code):
    """基准指数转JSON"""
    nav_norm = series / series.iloc[0]
    return {
        'name': name,
        'code': code,
        'nav': [
            {'date': d.strftime('%Y-%m-%d'), 'nav': round(float(v), 4)}
            for d, v in nav_norm.items()
        ],
    }


# ============================================================
# 主函数
# ============================================================
def main():
    os.makedirs(STRATEGY_DIR, exist_ok=True)
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    # 1. 拉取数据
    etf_data, benchmark_data = fetch_all_data()

    if len(etf_data) < 2:
        print("\n错误: 需要两只ETF数据才能回测!")
        sys.exit(1)

    # 2. 数据预处理
    prices = prepare_prices(etf_data)
    bench = prepare_benchmarks(benchmark_data) if benchmark_data else None

    print(f"\n对齐后交易日: {len(prices)}")
    print(f"日期范围: {prices.index[0].date()} ~ {prices.index[-1].date()}")
    print(f"ETF列: {prices.columns.tolist()}")

    # 3. 运行策略
    strategies = [
        ('动量轮动', 'momentum', strategy_momentum_rotation),
        ('等权再平衡', 'equal_weight', strategy_equal_weight),
        ('相对强弱动态配比', 'relative_strength', strategy_relative_strength),
    ]

    all_metrics = []
    for name, stype, func in strategies:
        print(f"\n[{name}]")
        returns, positions = func(prices)
        nav = calc_nav(returns)
        metrics = calc_metrics(returns, name)
        metrics['strategy_type'] = stype
        all_metrics.append(metrics)

        # 保存JSON
        jdata = strategy_to_json(nav, name, stype, metrics, positions)
        path = os.path.join(STRATEGY_DIR, f'{stype}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(jdata, f, ensure_ascii=False)
        print(f"  -> {path}")

    # 4. 保存基准
    if bench is not None:
        for code, info in BENCHMARK_CONFIG.items():
            name = info['name']
            if name in bench.columns:
                jdata = benchmark_to_json(bench[name], name, code)
                path = os.path.join(BENCHMARK_DIR, f"{info['file']}.json")
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(jdata, f, ensure_ascii=False)
                print(f"  基准 -> {path}")

    # 5. 汇总
    summary = {
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'etfs': {k: {kk: vv for kk, vv in v.items()} for k, v in ETF_CONFIG.items()},
        'benchmarks': {k: {kk: vv for kk, vv in v.items()} for k, v in BENCHMARK_CONFIG.items()},
        'benchmark_note': '偏股混合基金指数885001为Wind独家编制，免费数据源无法获取，改用标普500指数(us.INX)作为海外基准',
        'strategies': all_metrics,
        'data_range': {
            'start': prices.index[0].strftime('%Y-%m-%d'),
            'end': prices.index[-1].strftime('%Y-%m-%d'),
            'trading_days': len(prices),
        },
        'rebalance_freq': '周频（每周五）',
        'momentum_lookback': f'{MOMENTUM_LOOKBACK}周',
        'transaction_cost': f'{TRANSACTION_COST*100}%',
    }
    path = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 6. 打印汇总
    print("\n" + "=" * 60)
    print("回测结果汇总")
    print("=" * 60)
    df = pd.DataFrame(all_metrics)
    cols = ['name', 'total_return', 'annual_return', 'max_drawdown', 'sharpe', 'calmar', 'win_rate', 'volatility']
    print(df[cols].to_string(index=False))
    print(f"\n数据区间: {summary['data_range']['start']} ~ {summary['data_range']['end']} ({summary['data_range']['trading_days']}个交易日)")
    print(f"更新时间: {summary['update_time']}")


if __name__ == '__main__':
    main()
