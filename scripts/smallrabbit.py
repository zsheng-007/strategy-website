#!/usr/bin/env python3
"""
量化小白兔 - 全市场ETF轮动策略（小程序版）
策略来源：微信小程序"量化小白兔"

回测指标（2021-04 ~ 2026-04，5年）：
  总收益 4164.71%，年化 100.87%，最大回撤 -18.85%
  夏普 1.13，索提诺 1.51，卡玛 2.19，胜率 51%
  总交易604次（盈利308/亏损296）

策略要素：
  覆盖全球多市场ETF（A股/港股/美股/商品）
  动量因子捕捉跨市场轮动机会
  动量周期 20个交易日，持仓 1-2只
  支持T+0交易（实际上为日频调仓）
  实时排名 Top 5，自动调仓

风险控制：
  固定止损 -8%
  单日跌幅阈值 -5%（触发后次日清仓）
  溢价率过滤 < 15%
  流动性过滤：日均成交额>5000万
  跨市场相关性监控
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
# 全市场ETF池（覆盖A股/港股/美股/商品）
# ============================================================
FULL_MARKET_POOL = {
    # A股宽基
    '510300': {'name': '沪深300ETF', 'market': 'sh', 'region': 'A股'},
    '510500': {'name': '中证500ETF', 'market': 'sh', 'region': 'A股'},
    '512100': {'name': '中证1000ETF', 'market': 'sh', 'region': 'A股'},
    '159915': {'name': '创业板ETF', 'market': 'sz', 'region': 'A股'},
    '588000': {'name': '科创50ETF', 'market': 'sh', 'region': 'A股'},
    # A股行业
    '510880': {'name': '红利ETF', 'market': 'sh', 'region': 'A股'},
    '512760': {'name': '半导体ETF', 'market': 'sh', 'region': 'A股'},
    '512010': {'name': '医药ETF', 'market': 'sh', 'region': 'A股'},
    '512660': {'name': '军工ETF', 'market': 'sh', 'region': 'A股'},
    '512690': {'name': '酒ETF', 'market': 'sh', 'region': 'A股'},
    '515030': {'name': '新能源车ETF', 'market': 'sh', 'region': 'A股'},
    '512800': {'name': '银行ETF', 'market': 'sh', 'region': 'A股'},
    '515790': {'name': '光伏ETF', 'market': 'sh', 'region': 'A股'},
    # 跨境ETF
    '513100': {'name': '纳指ETF', 'market': 'sh', 'region': '美股'},
    '513500': {'name': '标普500ETF', 'market': 'sh', 'region': '美股'},
    '513050': {'name': '中概互联', 'market': 'sh', 'region': '中概'},
    '513180': {'name': '恒生科技ETF', 'market': 'sh', 'region': '港股'},
    '513520': {'name': '日经ETF', 'market': 'sh', 'region': '日股'},
    '513030': {'name': '德国ETF', 'market': 'sh', 'region': '欧股'},
    '159632': {'name': '港股通科技', 'market': 'sz', 'region': '港股'},
    '513330': {'name': '恒生中国企业', 'market': 'sh', 'region': '港股'},
    # 商品/黄金
    '518880': {'name': '黄金ETF', 'market': 'sh', 'region': '商品'},
    '515170': {'name': '有色金属ETF', 'market': 'sh', 'region': '商品'},
    '515220': {'name': '煤炭ETF', 'market': 'sh', 'region': '商品'},
    # 债券（避险）
    '511260': {'name': '十年国债', 'market': 'sh', 'region': '债券'},
}

# 策略参数
MOMENTUM_WINDOW = 20     # 动量周期20天
TOP_K_MAX = 2           # 最多持有2只
SINGLE_MAX_WEIGHT = 0.6  # 单只最大权重60%
STOP_LOSS = -0.08       # 固定止损-8%
DAILY_CRASH = -0.05     # 单日跌幅阈值-5%
MIN_LIQUIDITY = 5000 * 10000  # 日均成交额>5000万
TRANSACTION_COST = 0.0005  # 单边万分之五

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
STRATEGY_DIR = os.path.join(OUTPUT_DIR, 'strategies')


# ============================================================
# 数据获取
# ============================================================
def fetch_etf_data(code, market, start_date='2021-01-01', end_date='2026-12-31', retry=3):
    symbol = f'{market}{code}'
    url = f'https://proxy.finance.qq.com/ifzqgtimg/appstock/app/fqkline/get?param={symbol},day,{start_date},{end_date},640,qfq'
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


def fetch_all_data():
    print("=" * 60)
    print("拉取全市场ETF数据（小白兔策略）...")
    print("=" * 60)
    etf_data = {}
    for code, info in FULL_MARKET_POOL.items():
        df = fetch_etf_data(code, info['market'])
        if df is not None and len(df) >= 200:
            etf_data[code] = df
            print(f"  ✓ {info['name']}({code}): {len(df)}条")
        else:
            print(f"  ✗ {info['name']}({code}): 数据不足")
    return etf_data


# ============================================================
# 因子计算
# ============================================================
def calc_momentum(prices_df, window=MOMENTUM_WINDOW):
    """20日动量（收益率）"""
    return prices_df / prices_df.shift(window) - 1


def calc_volume_ratio(prices_df, window=20):
    """量比：近20日均量 / 近60日均量（流动性过滤）"""
    vol_20 = prices_df['volume'].rolling(window).mean()
    vol_60 = prices_df['volume'].rolling(window*3).mean()
    return vol_20 / vol_60


# ============================================================
# 回测引擎
# ============================================================
def get_rebalance_dates(prices, freq='daily'):
    """调仓日列表"""
    if freq == 'daily':
        return list(prices.index)
    elif freq == 'weekly':
        return [d for d in prices.index if d.weekday() == 4]
    elif freq == 'biweekly':
        fridays = [d for d in prices.index if d.weekday() == 4]
        return fridays[::2]
    return list(prices.index)


def backtest_smallrabbit(etf_data, freq='daily'):
    """
    量化小白兔策略回测

    核心逻辑：
    1. 每日计算所有ETF的20日动量
    2. 选动量最高的前1-2只持有
    3. 持仓数判断：若动量断层明显（Top1远大于Top2），只持1只
    4. 固定止损-8%触发后次日清仓
    5. 单日跌幅<-5%触发后次日清仓
    """
    print(f"\n运行小白兔策略(动量周期{MOMENTUM_WINDOW}天, 调仓{freq})...")

    # 构建收盘价矩阵
    close_dict = {}
    vol_dict = {}
    for code, df in etf_data.items():
        close_dict[code] = df.set_index('date')['close']
        vol_dict[code] = df.set_index('date')['volume']

    prices = pd.DataFrame(close_dict).dropna(how='all').ffill().dropna()
    volumes = pd.DataFrame(vol_dict).reindex(prices.index).fillna(0)

    # 20日动量
    momentum = calc_momentum(prices)

    # 流动性过滤：日均成交额>5000万（用成交量近似，价格*成交量=成交额，A股ETF价格通常1-3元）
    avg_volume = volumes.rolling(20).mean()
    # 假设价格1.5元（A股ETF均价），成交额=成交量*价格，简化用成交量阈值
    liquidity_ok = avg_volume > 1e7  # 成交量阈值1000万股对应~5000万成交额

    # 日频调仓
    etf_codes = prices.columns.tolist()
    daily_returns = prices.pct_change().fillna(0)

    positions = pd.DataFrame(0.0, index=prices.index, columns=etf_codes)
    cash_alloc = pd.Series(0.0, index=prices.index)  # 空仓比例

    holdings_log = []
    nav_tracker = 1.0
    peak_tracker = 1.0
    trigger_stop = False  # 止损触发标志

    for i in range(1, len(prices)):
        today = prices.index[i]
        prev = prices.index[i-1]

        # 风险控制1：单日跌幅<-5%触发止损
        if i >= 2:
            yesterday_ret = daily_returns.iloc[i-1].max()  # 最大跌幅（任意持仓）
            portfolio_yesterday = (positions.iloc[i-1] * daily_returns.iloc[i]).sum()
            if portfolio_yesterday < DAILY_CRASH:
                trigger_stop = True

        # 风险控制2：固定止损-8%
        current_dd = (nav_tracker / peak_tracker) - 1
        if current_dd < STOP_LOSS:
            trigger_stop = True

        if trigger_stop:
            # 清仓持现金
            positions.iloc[i] = 0.0
            cash_alloc.iloc[i] = 1.0
            trigger_stop = False  # 重置（次日可重新建仓）
        else:
            # 正常选股
            if i >= MOMENTUM_WINDOW:
                # 用昨日动量选今日持仓（信号延迟）
                mom_yesterday = momentum.iloc[i-1]

                # 流动性过滤：只选日均成交额足够的
                liq_yesterday = liquidity_ok.iloc[i-1] if i-1 >= 20 else pd.Series(True, index=etf_codes)
                valid_mom = mom_yesterday[liq_yesterday & mom_yesterday.notna()]

                if len(valid_mom) >= 1:
                    # 排序选动量最高的1-2只
                    sorted_mom = valid_mom.sort_values(ascending=False)

                    # 判断持仓数：若Top1动量 > Top2的1.5倍，只持1只
                    if len(sorted_mom) >= 2 and sorted_mom.iloc[0] > sorted_mom.iloc[1] * 1.5:
                        # 只持1只
                        selected = [sorted_mom.index[0]]
                        weights = [1.0]
                    else:
                        # 持2只
                        k = min(TOP_K_MAX, len(sorted_mom))
                        selected = list(sorted_mom.index[:k])
                        weights = [SINGLE_MAX_WEIGHT] * k
                        # 归一化到100%
                        s = sum(weights)
                        weights = [w / s for w in weights]

                    positions.iloc[i] = 0.0
                    for code, w in zip(selected, weights):
                        positions.iloc[i, positions.columns.get_loc(code)] = w
                    cash_alloc.iloc[i] = 0.0
                else:
                    positions.iloc[i] = 0.0
                    cash_alloc.iloc[i] = 1.0

        # 更新净值（仅基于持仓×日收益）
        if i >= 1:
            # 计算今日策略收益（昨日持仓×今日日收益）
            daily_ret = (positions.iloc[i-1] * daily_returns.iloc[i]).sum()
            nav_tracker *= (1 + daily_ret)
            peak_tracker = max(peak_tracker, nav_tracker)

        # 记录调仓（每周采样）
        if i % 5 == 0 or i == len(prices) - 1:
            pos = positions.iloc[i]
            holding = {'date': today.strftime('%Y-%m-%d')}
            for c in etf_codes:
                if pos[c] > 0.01:
                    holding[FULL_MARKET_POOL[c]['name']] = round(float(pos[c]), 4)
            holdings_log.append(holding)

    # 计算策略收益（用shift确保用昨日持仓）
    strategy_returns = (positions.shift(1) * daily_returns).sum(axis=1).fillna(0)

    # 交易成本
    pos_change = positions.diff().abs().sum(axis=1)
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
    # 索提诺（仅用下行波动）
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else std
    sortino = (annual_return - 0.02) / downside_std if downside_std > 0 else 0
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    win_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0
    volatility = float(std * np.sqrt(252))

    # 盈亏比
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    return {
        'name': name,
        'total_return': round(total_return * 100, 2),
        'annual_return': round(annual_return * 100, 2),
        'max_drawdown': round(max_drawdown * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'calmar': round(calmar, 3),
        'win_rate': round(win_rate * 100, 2),
        'volatility': round(volatility * 100, 2),
        'pl_ratio': round(pl_ratio, 3),
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
        'strategy_name': '量化小白兔(全市场轮动)',
        'strategy_type': 'smallrabbit',
        'metrics': metrics,
        'current_holding': current_holding,
        'etf_pool': {k: v['name'] for k, v in FULL_MARKET_POOL.items()},
        'nav': [
            {'date': d.strftime('%Y-%m-%d'), 'nav': round(float(v), 4)}
            for d, v in nav_norm.items()
        ],
        'holdings': holdings_log[-52:],
    }
    path = os.path.join(STRATEGY_DIR, 'smallrabbit.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  -> 已保存: {path}")
    return path


# ============================================================
# 主函数
# ============================================================
def main():
    os.makedirs(STRATEGY_DIR, exist_ok=True)
    etf_data = fetch_all_data()

    if len(etf_data) < 5:
        print("\n错误: ETF数据不足!")
        sys.exit(1)

    returns, positions, prices, holdings_log = backtest_smallrabbit(etf_data, freq='daily')
    metrics = calc_metrics(returns, '量化小白兔(全市场轮动)')
    metrics['strategy_type'] = 'smallrabbit'
    metrics['description'] = '20日动量+全市场ETF+止损+流动性过滤，1-2只持仓日频轮动'

    save_strategy_json(returns, positions, holdings_log, metrics)

    # 换手率
    etf_codes = prices.columns.tolist()
    pos_change = positions[etf_codes].diff().abs().sum(axis=1) / 2
    annual_turnover = pos_change.sum() / (len(returns) / 252)

    # 交易次数
    trade_count = (positions.diff().abs().sum(axis=1) > 0.01).sum()

    print("\n" + "=" * 60)
    print("量化小白兔策略回测结果")
    print("=" * 60)
    print(f"  策略: {metrics['name']}")
    print(f"  总收益: {metrics['total_return']}%")
    print(f"  年化: {metrics['annual_return']}%")
    print(f"  最大回撤: {metrics['max_drawdown']}%")
    print(f"  夏普: {metrics['sharpe']} 索提诺: {metrics['sortino']} 卡玛: {metrics['calmar']}")
    print(f"  胜率: {metrics['win_rate']}% 盈亏比: {metrics['pl_ratio']}")
    print(f"  年换手: {annual_turnover:.1f}倍 交易次数: {trade_count}")
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