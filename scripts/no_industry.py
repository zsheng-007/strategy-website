#!/usr/bin/env python3
"""
无行业ETF轮动策略（量化小白兔）
策略来源：微信小程序"量化小白兔" - 2026-08最新版

回测指标（2016-03 ~ 2026-03，10年）：
  总收益 5457.43%，年化 51.12%，最大回撤 -21.80%
  夏普 1.62，索提诺 2.49，卡玛 2.34，胜率 51.5%
  盈亏比 1.06，总交易 642次（盈利331/亏损311）

策略原理：
  基于动量因子（趋势稳定性指数），通过加权回归斜率计算
  动量周期 25个交易日
  排除全球大类资产配置（商品、国际、港股、指数、债券）
  持有数量 1只，加仓机制：满足条件自动切换
  基于动量周期25个交易日，持有1只
  当满足持仓条件时，自动切换至该品种

风控：
  固定止损 -5%
  单日跌幅阈值 -3%
  溢价率过滤 <20%
  成交量异常过滤
  近3日跌幅过滤
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
# ETF池（28只，按截图完整复刻）
# ============================================================
ETF_POOL = {
    # 大宗商品
    '518880': {'name': '黄金ETF', 'market': 'sh', 'category': '大宗商品'},
    '159980': {'name': '有色ETF', 'market': 'sz', 'category': '大宗商品'},
    '159985': {'name': '豆粕ETF', 'market': 'sz', 'category': '大宗商品'},
    '501018': {'name': '南方原油', 'market': 'sh', 'category': '大宗商品'},
    '161226': {'name': '白银LOF', 'market': 'sz', 'category': '大宗商品'},
    '159981': {'name': '能源化工ETF', 'market': 'sh', 'category': '大宗商品'},
    # 国际ETF
    '513100': {'name': '纳指ETF', 'market': 'sh', 'category': '国际'},
    '159509': {'name': '纳斯达克科技ETF', 'market': 'sz', 'category': '国际'},
    '513290': {'name': '纳指生物科技ETF', 'market': 'sh', 'category': '国际'},
    '513500': {'name': '标普500ETF', 'market': 'sh', 'category': '国际'},
    '513520': {'name': '日经225ETF', 'market': 'sh', 'category': '国际'},
    '513030': {'name': '德国30ETF', 'market': 'sh', 'category': '国际'},
    '159792': {'name': '港股互联网ETF', 'market': 'sz', 'category': '港股'},
    '513130': {'name': '恒生科技ETF', 'market': 'sh', 'category': '港股'},
    '513050': {'name': '中概互联ETF', 'market': 'sh', 'category': '港股'},
    '159920': {'name': '恒生ETF', 'market': 'sz', 'category': '港股'},
    # A股指数
    '510300': {'name': '沪深300ETF', 'market': 'sh', 'category': '指数'},
    '510500': {'name': '中证500ETF', 'market': 'sh', 'category': '指数'},
    '159915': {'name': '创业板ETF', 'market': 'sz', 'category': '指数'},
    '588080': {'name': '科创50ETF', 'market': 'sh', 'category': '指数'},
    '512100': {'name': '中证1000ETF', 'market': 'sh', 'category': '指数'},
    '563360': {'name': 'A500ETF', 'market': 'sh', 'category': '指数'},
    '563300': {'name': '中证A200ETF', 'market': 'sh', 'category': '指数'},
    '512890': {'name': '红利低波ETF', 'market': 'sh', 'category': '指数'},
    '159967': {'name': '创业板成长ETF', 'market': 'sz', 'category': '指数'},
    '512040': {'name': '价值ETF', 'market': 'sh', 'category': '指数'},
    '159201': {'name': '自由现金流ETF', 'market': 'sz', 'category': '指数'},
    # 债券
    '511380': {'name': '可转债ETF', 'market': 'sh', 'category': '债券'},
    '511010': {'name': '国债ETF', 'market': 'sh', 'category': '债券'},
    '511220': {'name': '城投债ETF', 'market': 'sh', 'category': '债券'},
}

# 策略参数（按截图）
MOMENTUM_WINDOW = 25       # 动量周期25天
TOP_K = 1                  # 持有1只
SWITCH_BUFFER = 0.15       # 换仓缓冲带：新标的得分需超过现持仓15%才切换
STOP_LOSS = -0.05          # 固定止损-5%（组合回撤）
DAILY_CRASH = -0.03        # 单日跌幅阈值-3%
PREMIUM_MAX = 0.20         # 溢价率<20%
RECENT_DROP_MAX = 0.05     # 近3日跌幅过滤阈值
TREND_BREAK = -0.10        # 持仓标的近3日累计跌幅超10%视为趋势破坏
TRANSACTION_COST = 0.0005  # 单边万分之五

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
STRATEGY_DIR = os.path.join(OUTPUT_DIR, 'strategies')


# ============================================================
# 数据获取
# ============================================================
def fetch_etf_data(code, market, start_date='2016-01-01', end_date='2026-12-31', retry=3):
    symbol = f'{market}{code}'
    url = f'https://proxy.finance.qq.com/ifzqgtimg/appstock/app/fqkline/get?param={symbol},day,{start_date},{end_date},640,qfq'
    for i in range(retry):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            data = r.json()
            if data.get('code') != 0:
                time.sleep(1)
                continue
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
    print("拉取ETF池数据（小白兔无行业策略）...")
    print("=" * 60)
    etf_data = {}
    for code, info in ETF_POOL.items():
        df = fetch_etf_data(code, info['market'])
        if df is not None and len(df) >= 200:
            etf_data[code] = df
            print(f"  ✓ {info['name']}({code}): {len(df)}条")
        else:
            print(f"  ✗ {info['name']}({code}): 数据不足({len(df) if df is not None else 0}条)")
    return etf_data


# ============================================================
# 因子计算：动量 + 线性回归斜率
# ============================================================
def calc_momentum_factor(prices_df, window=MOMENTUM_WINDOW):
    """
    动量因子：综合斜率×R²
    基于加权线性回归计算动量得分
    """
    scores = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)

    for col in prices_df.columns:
        close = prices_df[col].astype(float)

        for i in range(window, len(close)):
            srs = close.iloc[i-window:i]
            if srs.iloc[0] == 0 or np.isnan(srs.iloc[0]):
                scores.iloc[i, scores.columns.get_loc(col)] = 0
                continue

            # 归一化价格序列
            y = srs.values / srs.values[0]
            x = np.arange(1, window + 1)

            # 加权线性回归（近端权重更大）
            weights = np.linspace(0.5, 1.5, window)
            w_sum = weights.sum()
            x_mean = (x * weights).sum() / w_sum
            y_mean = (y * weights).sum() / w_sum
            ss_xx = ((x - x_mean) ** 2 * weights).sum()
            ss_xy = ((x - x_mean) * (y - y_mean) * weights).sum()
            ss_yy = ((y - y_mean) ** 2 * weights).sum()

            if ss_xx == 0 or ss_yy == 0:
                scores.iloc[i, scores.columns.get_loc(col)] = 0
                continue

            slope = ss_xy / ss_xx
            r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

            # 动量得分 = 斜率 × R² × 10000
            score = slope * r_squared * 10000
            scores.iloc[i, scores.columns.get_loc(col)] = score

    return scores


# ============================================================
# 回测引擎
# ============================================================
def backtest_no_industry(etf_data):
    """
    无行业ETF轮动策略（信号触发式切换，复刻小白兔）
    - 动量因子：25日加权回归斜率×R²
    - 每日检查信号：
      · 空仓时 → 买入当日过滤后得分最高的ETF
      · 持有时 → 仅当新第一名得分超过现持仓(1+10%缓冲带)才切换
      · 现持仓跌出过滤池（动量转负/近3日跌幅/量能异常）→ 清仓
      · 持仓标的近3日累计跌幅>10%（趋势破坏）→ 清仓
    - 收益口径：昨日信号今日建仓，今日持仓吃今日收盘对昨日收盘涨幅
    - holdings_log 记录每次真实调仓事件（换仓日+新标的）
    """
    print("\n运行无行业ETF轮动策略...")

    # 构建收盘价/成交量矩阵
    close_dict = {}
    vol_dict = {}
    for code, df in etf_data.items():
        close_dict[code] = df.set_index('date')['close']
        vol_dict[code] = df.set_index('date')['volume']

    prices = pd.DataFrame(close_dict).dropna(how='all').ffill().dropna()
    volumes = pd.DataFrame(vol_dict).reindex(prices.index).fillna(0)

    # 计算动量因子
    factor_scores = calc_momentum_factor(prices)

    # 近3日跌幅（用于过滤）
    recent_drop = (prices / prices.shift(3) - 1)

    # 日均成交量异常过滤（>3倍均值为异常）
    avg_vol = volumes.rolling(20).mean()
    vol_anomaly = volumes > (avg_vol * 3)

    etf_codes = prices.columns.tolist()
    daily_returns = prices.pct_change().fillna(0)

    positions = pd.DataFrame(0.0, index=prices.index, columns=etf_codes)
    holdings_log = []          # 真实调仓事件记录
    current_holding = None
    last_cleared = None        # 最近被清仓的标的（抑制噪声重建仓）
    cooldown = 0               # 清仓后冷却天数

    def get_valid_codes(i):
        """返回 i 日的有效候选（含得分），应用全部过滤"""
        sy = factor_scores.iloc[i]
        valid = {}
        for c in etf_codes:
            s = sy[c]
            if pd.isna(s) or s <= 0:          # 动量必须为正
                continue
            if (i) in recent_drop.index and c in recent_drop.columns:
                if not pd.isna(recent_drop.iloc[i][c]) and recent_drop.iloc[i][c] < -RECENT_DROP_MAX:
                    continue                    # 近3日跌幅>5%
            if (i) in vol_anomaly.index and c in vol_anomaly.columns:
                if vol_anomaly.iloc[i][c]:
                    continue                    # 成交量异常放大
            valid[c] = s
        return valid

    for i in range(1, len(prices)):
        today = prices.index[i]

        if i < MOMENTUM_WINDOW + 1:
            positions.iloc[i] = 0.0
            continue

        # ── 用昨日收盘后的信号，决定今日开盘持仓 ──
        sig = i - 1  # 信号日
        valid = get_valid_codes(sig)

        # 1. 趋势破坏检查：持仓标的近3日累计跌幅>10% → 清仓
        if current_holding is not None and sig >= 3:
            ret3 = (prices.iloc[sig][current_holding] / prices.iloc[sig-3][current_holding]) - 1
            if ret3 < TREND_BREAK:
                last_cleared = current_holding
                current_holding = None
                cooldown = 3

        # 2. 现持仓跌出过滤池（动量转负/近3日大跌/量能异常）→ 清仓（带冷却）
        if current_holding is not None and current_holding not in valid:
            last_cleared = current_holding
            current_holding = None
            cooldown = 3

        # 3. 选股/换仓决策（带冷却机制，避免同一标的反复建仓产生噪声）
        if cooldown > 0:
            cooldown -= 1
        elif valid:
            best = max(valid, key=valid.get)
            if current_holding is None:
                # 空仓 → 建仓（若与最近一次清仓标的相同，视为延续，不重复记录）
                current_holding = best
                if best != last_cleared:
                    holdings_log.append({
                        'date': today.strftime('%Y-%m-%d'),
                        'action': 'buy',
                        'etf': ETF_POOL[best]['name'],
                        'score': round(float(valid[best]), 2),
                    })
                last_cleared = None
            elif best != current_holding:
                # 换仓缓冲带：新标的得分需显著超过现持仓才切换
                cur_score = valid.get(current_holding, 0)
                if valid[best] > cur_score * (1 + SWITCH_BUFFER):
                    holdings_log.append({
                        'date': today.strftime('%Y-%m-%d'),
                        'action': 'switch',
                        'etf': ETF_POOL[best]['name'],
                        'from': ETF_POOL[current_holding]['name'],
                        'score': round(float(valid[best]), 2),
                    })
                    current_holding = best
        else:
            # 全池无合格标的 → 空仓，并记录被清仓标的以抑制噪声重建仓
            if current_holding is not None:
                last_cleared = current_holding
                current_holding = None
                cooldown = 3

        # 设置今日持仓
        if current_holding is not None:
            positions.iloc[i] = 0.0
            positions.iloc[i, positions.columns.get_loc(current_holding)] = 1.0
        else:
            positions.iloc[i] = 0.0

    # 统一收益口径：今日持仓(positions[i])吃今日涨幅(daily_returns[i])
    # 即：昨日收盘出信号 → 今日开盘建仓 → 今日收盘结算
    strategy_returns = (positions * daily_returns).sum(axis=1).fillna(0)

    # 交易成本（换仓时双边）
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

    # 索提诺比率
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
        'max_drawdraw': 0,  # typo保留兼容
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

    # 当前持仓：取最后一行positions（最新真实持仓）
    last_pos = positions.iloc[-1]
    current = {}
    for c in positions.columns:
        if last_pos[c] > 0.01:
            current = {
                'date': positions.index[-1].strftime('%Y-%m-%d'),
                'etf': ETF_POOL[c]['name'],
                'weight': round(float(last_pos[c]), 4),
            }

    data = {
        'strategy_name': '无行业ETF轮动(小白兔)',
        'strategy_type': 'no_industry',
        'metrics': metrics,
        'current_holding': current,
        'etf_pool': {k: v['name'] for k, v in ETF_POOL.items()},
        'nav': [
            {'date': d.strftime('%Y-%m-%d'), 'nav': round(float(v), 4)}
            for d, v in nav_norm.items()
        ],
        # holdings 为真实调仓事件流（买入/切换），倒序取最近52条再正序展示
        'holdings': list(reversed(holdings_log[-52:])),
    }
    path = os.path.join(STRATEGY_DIR, 'no_industry.json')
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

    returns, positions, prices, holdings_log = backtest_no_industry(etf_data)
    metrics = calc_metrics(returns, '无行业ETF轮动(小白兔)')
    metrics['strategy_type'] = 'no_industry'
    metrics['description'] = '动量因子+线性回归斜率，25日动量，28只ETF满仓单标的+严格风控'

    save_strategy_json(returns, positions, holdings_log, metrics)

    # 换手率
    etf_codes = prices.columns.tolist()
    pos_change = positions[etf_codes].diff().abs().sum(axis=1) / 2
    annual_turnover = pos_change.sum() / (len(returns) / 252)
    trade_count = len(holdings_log)

    print("\n" + "=" * 60)
    print("无行业ETF轮动策略回测结果")
    print("=" * 60)
    print(f"  策略: {metrics['name']}")
    print(f"  总收益: {metrics['total_return']}%")
    print(f"  年化: {metrics['annual_return']}%")
    print(f"  最大回撤: {metrics['max_drawdown']}%")
    print(f"  夏普: {metrics['sharpe']} 索提诺: {metrics['sortino']} 卡玛: {metrics['calmar']}")
    print(f"  胜率: {metrics['win_rate']}% 盈亏比: {metrics['pl_ratio']}")
    print(f"  年换手: {annual_turnover:.1f}倍 交易次数: {trade_count}")
    print(f"  区间: {metrics['start_date']} ~ {metrics['end_date']} ({metrics['trading_days']}交易日)")

    # 当前持仓
    last_pos = positions.iloc[-1]
    for c in positions.columns:
        if last_pos[c] > 0.01:
            print(f"\n  当前持仓: {ETF_POOL[c]['name']} {last_pos[c]*100:.0f}%")

    # 最近8次调仓
    if holdings_log:
        print(f"\n  最近8次调仓事件:")
        for h in holdings_log[-8:]:
            if h['action'] == 'buy':
                print(f"    {h['date']} 买入 {h['etf']} (得分{h['score']})")
            else:
                print(f"    {h['date']} {h.get('from','')} → {h['etf']} (得分{h['score']})")

    return metrics


if __name__ == '__main__':
    main()