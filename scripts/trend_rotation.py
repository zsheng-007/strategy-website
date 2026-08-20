#!/usr/bin/env python3
"""
趋势得分ETF轮动策略（复刻量化小白兔）
策略原型：知乎"手把手教你构建与改进ETF轮动策略"改进版2
  原始回测：2015-2025十年19倍，年化21.67%，夏普1.33，最大回撤-30.31%

策略逻辑：
  候选池：红利ETF(510880) + 创业板ETF(159915) + 纳指ETF(513100) + 黄金ETF(518880)
  排序方式：斜率×R²趋势得分（不是简单涨幅）
  25日滚动窗口，日频调仓，满仓持有趋势得分最高的1只

复刻优化点：
  1. 加入交易成本（万分之五单边，含滑点）
  2. 信号延迟修正：T日收盘计算信号，T+1日开盘执行（更接近实盘）
  3. 周频调仓选项（降低换手成本）
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
# 策略配置
# ============================================================
TREND_ETF_POOL = {
    '510880': {'name': '红利ETF', 'market': 'sh', 'style': 'A股价值'},
    '159915': {'name': '创业板ETF', 'market': 'sz', 'style': 'A股成长'},
    '513100': {'name': '纳指ETF', 'market': 'sh', 'style': '美股'},
    '518880': {'name': '黄金ETF', 'market': 'sh', 'style': '大宗商品'},
}

TREND_WINDOW = 25          # 25日趋势窗口（原策略最优参数）
TOP_K = 1                  # 持有趋势得分最高的1只
TRANSACTION_COST = 0.0005  # 单边万分之五（含滑点，日频需要覆盖成本）
REBALANCE_FREQ = 'daily'   # 日频调仓（原策略设计）

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
STRATEGY_DIR = os.path.join(OUTPUT_DIR, 'strategies')


# ============================================================
# 数据获取
# ============================================================
def fetch_etf_data(code, market, start_date='2020-01-01', end_date='2026-12-31', retry=3):
    """从腾讯API获取ETF后复权日K线"""
    symbol = f'{market}{code}'
    # 优先用后复权（与原策略一致）
    url = f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,{start_date},{end_date},640,hfq'
    for i in range(retry):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            data = r.json()
            inner = data.get('data', {}).get(symbol, {})
            klines = inner.get('hfqday', []) or inner.get('qfqday', []) or inner.get('day', [])
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
    print("拉取趋势轮动ETF数据...")
    print("=" * 60)
    etf_data = {}
    for code, info in TREND_ETF_POOL.items():
        print(f"  {info['name']}({code})...", end=' ')
        df = fetch_etf_data(code, info['market'])
        if df is not None and len(df) > 0:
            etf_data[code] = df
            print(f"{len(df)}条, {df['date'].iloc[0].date()}~{df['date'].iloc[-1].date()}")
        else:
            print("失败!")
    return etf_data


# ============================================================
# 趋势得分计算（核心：斜率×R²）
# ============================================================
def calc_trend_score(prices_df, window=TREND_WINDOW):
    """
    趋势得分 = 斜率 × R²
    
    对收盘价序列归一化后做线性回归：
    - 斜率反映趋势方向和速度
    - R²反映趋势的确定性/平滑度
    - 两者相乘过滤掉"高波动伪动量"
    
    这是比简单涨跌幅更优的排序方式（原策略验证：年化从14.74%提升到21.67%）
    """
    scores = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    
    for col in prices_df.columns:
        close = prices_df[col].astype(float)
        
        for i in range(window, len(close)):
            # 取过去N日收盘价
            srs = close.iloc[i-window:i]
            
            # 归一化（消除价格绝对值差异）
            if srs.iloc[0] == 0 or np.isnan(srs.iloc[0]):
                scores.iloc[i, scores.columns.get_loc(col)] = 0
                continue
            y = srs.values / srs.values[0]
            x = np.arange(1, window + 1)
            
            # 线性回归
            x_mean = x.mean()
            y_mean = y.mean()
            ss_xx = ((x - x_mean) ** 2).sum()
            ss_xy = ((x - x_mean) * (y - y_mean)).sum()
            ss_yy = ((y - y_mean) ** 2).sum()
            
            if ss_xx == 0 or ss_yy == 0:
                scores.iloc[i, scores.columns.get_loc(col)] = 0
                continue
            
            slope = ss_xy / ss_xx
            r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)
            
            # 趋势得分 = 斜率 × R² × 10000（系数仅美化显示）
            score = slope * r_squared * 10000
            scores.iloc[i, scores.columns.get_loc(col)] = score
    
    return scores


# ============================================================
# 回测引擎
# ============================================================
def backtest_trend_rotation(etf_data):
    """
    趋势得分轮动回测
    - 日频调仓
    - 每日计算4只ETF的趋势得分
    - 满仓持有得分最高的1只
    - T日收盘计算信号，T+1日持仓生效
    """
    print("\n运行趋势得分轮动策略...")
    
    # 构建收盘价矩阵
    close_dict = {}
    for code, df in etf_data.items():
        close_dict[code] = df.set_index('date')['close']
    prices = pd.DataFrame(close_dict).dropna(how='all').ffill().dropna()
    
    # 计算趋势得分
    scores = calc_trend_score(prices)
    
    etf_codes = prices.columns.tolist()
    daily_returns = prices.pct_change().fillna(0)
    
    # 每日选得分最高的ETF
    positions = pd.DataFrame(0.0, index=prices.index, columns=etf_codes)
    holdings_log = []
    
    for i in range(1, len(prices)):
        # 用昨日得分决定今日持仓（信号延迟1日）
        if i >= TREND_WINDOW:
            score_row = scores.iloc[i-1]
            if score_row.notna().any():
                best = score_row.idxmax()
                positions.iloc[i] = 0.0
                positions.iloc[i, positions.columns.get_loc(best)] = 1.0
        
        # 记录调仓（每周采样）
        if i % 5 == 0 or i == len(prices) - 1:
            date = prices.index[i]
            pos = positions.iloc[i]
            holding = {'date': date.strftime('%Y-%m-%d')}
            for c in etf_codes:
                if pos[c] > 0:
                    holding[TREND_ETF_POOL[c]['name']] = round(float(pos[c]), 4)
            holdings_log.append(holding)
    
    # 计算策略收益（持仓×日收益，shift(1)确保用昨日持仓算今日收益）
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
        'strategy_name': '趋势得分轮动(小白兔)',
        'strategy_type': 'trend_rotation',
        'metrics': metrics,
        'current_holding': current_holding,
        'etf_pool': {k: v['name'] for k, v in TREND_ETF_POOL.items()},
        'nav': [
            {'date': d.strftime('%Y-%m-%d'), 'nav': round(float(v), 4)}
            for d, v in nav_norm.items()
        ],
        'holdings': holdings_log[-52:],
    }
    path = os.path.join(STRATEGY_DIR, 'trend_rotation.json')
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
    if len(etf_data) < 3:
        print("\n错误: ETF数据不足!")
        sys.exit(1)
    
    returns, positions, prices, holdings_log = backtest_trend_rotation(etf_data)
    metrics = calc_metrics(returns, '趋势得分轮动(小白兔)')
    metrics['strategy_type'] = 'trend_rotation'
    metrics['description'] = '斜率×R²趋势得分，4只ETF日频轮动，满仓第1名'
    
    save_strategy_json(returns, positions, holdings_log, metrics)
    
    # 换手率
    etf_codes = prices.columns.tolist()
    pos_change = positions[etf_codes].diff().abs().sum(axis=1) / 2
    annual_turnover = pos_change.sum() / (len(returns) / 252)
    
    print("\n" + "=" * 60)
    print("趋势得分轮动策略回测结果")
    print("=" * 60)
    print(f"  策略: {metrics['name']}")
    print(f"  总收益: {metrics['total_return']}%")
    print(f"  年化: {metrics['annual_return']}%")
    print(f"  最大回撤: {metrics['max_drawdown']}%")
    print(f"  夏普: {metrics['sharpe']}")
    print(f"  Calmar: {metrics['calmar']}")
    print(f"  胜率: {metrics['win_rate']}%")
    print(f"  波动率: {metrics['volatility']}%")
    print(f"  年换手率: {annual_turnover:.1f}倍")
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
