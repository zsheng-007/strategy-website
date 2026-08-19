#!/usr/bin/env python3
"""
多元配置策略 - 全天候全球资产配置
策略原型：广发金工全天候多元配置ETF组合（年化9.22%，最大回撤3.64%，夏普~2.4）
简化实现：风险平价模型 + 动量过滤，利用跨境ETF实现全球分散配置

标的池：A股宽基 + 跨境ETF（美股/日股/港股/欧洲/中概/商品）
配置模型：风险平价（按波动率反比分配，各资产风险贡献相等）
再平衡频率：月频
动态调整：12月动量过滤（剔除趋势向下标的）
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
# 全球多元资产ETF池
# ============================================================
GLOBAL_ETF_POOL = {
    # A股宽基
    '510300': {'name': '沪深300ETF', 'market': 'sh', 'region': 'A股', 'class': '股票'},
    '159915': {'name': '创业板ETF', 'market': 'sz', 'region': 'A股', 'class': '股票'},
    # 跨境ETF - 美股
    '513100': {'name': '纳指ETF', 'market': 'sh', 'region': '美国', 'class': '股票'},
    '513500': {'name': '标普500ETF', 'market': 'sh', 'region': '美国', 'class': '股票'},
    '513400': {'name': '道琼斯ETF', 'market': 'sh', 'region': '美国', 'class': '股票'},
    # 跨境ETF - 港股（精简为2只，去掉重复）
    '513180': {'name': '恒生科技ETF', 'market': 'sh', 'region': '中国香港', 'class': '股票'},
    '159632': {'name': '港股通科技ETF', 'market': 'sz', 'region': '中国香港', 'class': '股票'},
    # 跨境ETF - 日本
    '513520': {'name': '日经ETF', 'market': 'sh', 'region': '日本', 'class': '股票'},
    # 跨境ETF - 欧洲
    '513030': {'name': '德国ETF', 'market': 'sh', 'region': '德国', 'class': '股票'},
    '513980': {'name': '法国CAC40ETF', 'market': 'sh', 'region': '法国', 'class': '股票'},
    # 跨境ETF - 新兴市场
    '164824': {'name': '印度基金LOF', 'market': 'sz', 'region': '印度', 'class': '股票'},
    '159100': {'name': '巴西ETF', 'market': 'sz', 'region': '巴西', 'class': '股票'},
    # 商品ETF - 黄金
    '518880': {'name': '黄金ETF', 'market': 'sh', 'region': '全球', 'class': '商品'},
    # 商品ETF - 有色金属
    '515170': {'name': '有色金属ETF', 'market': 'sh', 'region': 'A股', 'class': '商品'},
    # 商品ETF - 能源化工
    '159981': {'name': '能源化工ETF', 'market': 'sh', 'region': 'A股', 'class': '商品'},
    # 商品ETF - 煤炭
    '515220': {'name': '煤炭ETF', 'market': 'sh', 'region': 'A股', 'class': '商品'},
    # 债券ETF - 国债（低波动，降低组合回撤）
    '511260': {'name': '十年国债ETF', 'market': 'sh', 'region': '中国', 'class': '债券'},
    # 货币ETF - 现金管理（替代空仓时的现金）
    '511990': {'name': '华宝添益', 'market': 'sh', 'region': '中国', 'class': '货币'},
}

REBALANCE_WEEKDAY = 4  # 周五（月频的第一个周五）
MOMENTUM_LOOKBACK = 60  # 60日动量（约3个月）
VOLATILITY_WINDOW = 60  # 60日波动率
TRANSACTION_COST = 0.0004  # 单边万分之四（跨境ETF流动性好但有时差滑点）
MIN_WEIGHT = 0.03  # 最小权重3%
MAX_WEIGHT = 0.30  # 最大权重30%
WEIGHT_CLIP = 0.20  # 单只权重上限20%（降低集中度）

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
STRATEGY_DIR = os.path.join(OUTPUT_DIR, 'strategies')


# ============================================================
# 数据获取
# ============================================================
def fetch_etf_data(code, market, start_date='2020-01-01', end_date='2026-12-31', retry=3):
    """从腾讯API获取ETF前复权日K线"""
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
    """拉取全部ETF数据"""
    print("=" * 60)
    print("拉取全球多元资产ETF数据...")
    print("=" * 60)

    etf_data = {}
    for code, info in GLOBAL_ETF_POOL.items():
        print(f"  {info['name']}({code})...", end=' ')
        df = fetch_etf_data(code, info['market'])
        if df is not None and len(df) > 0:
            etf_data[code] = df
            print(f"{len(df)}条, {df['date'].iloc[0].date()}~{df['date'].iloc[-1].date()}")
        else:
            print("失败!")

    return etf_data


# ============================================================
# 风险平价模型
# ============================================================
def calc_volatility(prices_df, window=VOLATILITY_WINDOW):
    """计算年化波动率"""
    returns = prices_df.pct_change()
    vol = returns.rolling(window=window).std() * np.sqrt(252)
    return vol


def risk_parity_weights(vol_row, cov_matrix=None):
    """
    风险平价权重分配（考虑相关性）
    使用迭代法求解：使每个资产的风险贡献相等
    若cov_matrix为None则退化为1/波动率方法
    """
    valid_vol = vol_row.dropna()
    if len(valid_vol) == 0:
        return pd.Series(0, index=vol_row.index)

    valid_codes = valid_vol.index.tolist()

    if cov_matrix is not None and len(cov_matrix) > 0:
        # 考虑协方差矩阵的完整风险平价（迭代法）
        cov = cov_matrix.loc[valid_codes, valid_codes].values
        n = len(valid_codes)

        # 初始权重：1/波动率
        vols = np.array([valid_vol[c] for c in valid_codes])
        w = (1.0 / vols)
        w = w / w.sum()

        # 迭代优化：使风险贡献相等
        for _ in range(100):
            # 计算各资产风险贡献
            portfolio_var = w @ cov @ w
            if portfolio_var <= 0:
                break
            marginal_contrib = cov @ w
            risk_contrib = w * marginal_contrib
            target_contrib = portfolio_var / n  # 等贡献目标

            # 调整权重
            adj = np.sqrt(target_contrib / np.maximum(risk_contrib, 1e-10))
            w = w * adj
            w = w / w.sum()

        weights = pd.Series(w, index=valid_codes)
    else:
        # 退化：1/波动率方法
        inv_vol = 1.0 / valid_vol
        weights = inv_vol / inv_vol.sum()

    # 裁剪极端权重
    weights = weights.clip(upper=WEIGHT_CLIP)
    weights = weights / weights.sum()

    # 最小权重
    for code in weights.index:
        if weights[code] < MIN_WEIGHT:
            weights[code] = MIN_WEIGHT
    weights = weights / weights.sum()

    # 填充
    full_weights = pd.Series(0.0, index=vol_row.index)
    full_weights[weights.index] = weights
    return full_weights

    # 设置最小权重
    for code in weights.index:
        if weights[code] < MIN_WEIGHT:
            weights[code] = MIN_WEIGHT
    weights = weights / weights.sum()

    # 填充NaN为0
    full_weights = pd.Series(0.0, index=vol_row.index)
    full_weights[weights.index] = weights

    return full_weights


# ============================================================
# 动量过滤
# ============================================================
def calc_momentum(prices_df, window=MOMENTUM_LOOKBACK):
    """计算过去N日动量（收益率）"""
    momentum = prices_df / prices_df.shift(window) - 1
    return momentum


# ============================================================
# 回测引擎
# ============================================================
def get_monthly_rebalance_dates(prices):
    """获取月频调仓日（每月第一个周五或月末）"""
    dates = []
    last_month = None
    for d in prices.index:
        if d.month != last_month:
            # 当月第一个交易日
            dates.append(d)
            last_month = d.month
    # 包含最后一天
    if prices.index[-1] not in dates:
        dates.append(prices.index[-1])
    return dates


def backtest_multi_asset(etf_data):
    """
    多元配置策略回测
    - 月频调仓
    - 风险平价模型分配权重
    - 动量过滤：剔除过去60日收益为负的标的
    """
    print("\n运行多元配置策略(风险平价)...")

    # 构建收盘价矩阵，过滤数据不足的标的（<200天的自动跳过，数据充足后自动纳入）
    MIN_DATA_POINTS = 200
    close_dict = {}
    skipped = []
    for code, df in etf_data.items():
        if len(df) >= MIN_DATA_POINTS:
            close_dict[code] = df.set_index('date')['close']
        else:
            skipped.append(f"{GLOBAL_ETF_POOL[code]['name']}({code}, {len(df)}天)")
    if skipped:
        print(f"  跳过数据不足的标的: {', '.join(skipped)}")
    prices = pd.DataFrame(close_dict).dropna(how='all').ffill().dropna()

    # 计算波动率和动量
    vol_df = calc_volatility(prices)
    mom_df = calc_momentum(prices)
    # 计算滚动协方差矩阵（用于考虑相关性的风险平价）
    daily_returns_raw = prices.pct_change().fillna(0)
    cov_window = VOLATILITY_WINDOW  # 60日协方差窗口

    # 获取调仓日（月频）
    rebal_dates = get_monthly_rebalance_dates(prices)

    etf_codes = prices.columns.tolist()
    positions = pd.DataFrame(0.0, index=prices.index, columns=etf_codes + ['现金'])
    daily_returns = prices.pct_change().fillna(0)

    current_pos = {c: 0.0 for c in etf_codes}
    current_pos['现金'] = 1.0  # 初始全仓现金

    holdings_log = []

    for i in range(len(rebal_dates) - 1):
        date = rebal_dates[i]
        next_date = rebal_dates[i + 1]

        if i >= 3:  # 至少3个月后开始
            vol_row = vol_df.loc[date]
            mom_row = mom_df.loc[date]

            # 动量过滤：只保留动量>0的标的（债券/货币不参与动量过滤）
            BOND_CODES = ['511260', '511990']  # 债券和货币ETF
            equity_codes = [c for c in vol_row.index if c not in BOND_CODES]
            valid_mask = (mom_row > 0) & vol_row.notna() & (vol_row > 0) & pd.Series(True, index=vol_row.index)
            # 只有权益/商品做动量过滤
            for c in BOND_CODES:
                if c in valid_mask.index:
                    valid_mask[c] = False  # 债券不参与动量筛选
            valid_vol = vol_row[valid_mask]

            # 固定债券配置比例（walk-forward最优参数）
            BOND_ALLOC = 0.05  # 国债5%
            CASH_ALLOC = 0.05  # 货币5%
            EQUITY_ALLOC = 1.0 - BOND_ALLOC - CASH_ALLOC  # 权益/商品90%

            if len(valid_vol) >= 2:
                # 计算当日协方差矩阵（考虑相关性）
                ret_up_to_date = daily_returns_raw.loc[:date]
                if len(ret_up_to_date) > cov_window:
                    cov_matrix = ret_up_to_date.iloc[-cov_window:].cov() * 252
                else:
                    cov_matrix = None

                # 风险平价权重（在权益/商品标的内分配85%）
                weights = risk_parity_weights(vol_row[valid_mask], cov_matrix)
                weights = weights[weights > 0] * EQUITY_ALLOC  # 按比例缩放

                current_pos = {c: 0.0 for c in etf_codes}
                current_pos['现金'] = 0.0
                for code in weights.index:
                    current_pos[code] = float(weights[code])
                # 固定配置债券和货币
                if '511260' in etf_codes:
                    current_pos['511260'] = BOND_ALLOC
                if '511990' in etf_codes:
                    current_pos['511990'] = CASH_ALLOC
            elif len(valid_vol) >= 1:
                current_pos = {c: 0.0 for c in etf_codes}
                current_pos['现金'] = 0.5
                for code in selected_codes:
                    current_pos[code] = 0.5 / len(selected_codes)
                for code in valid_vol.index:
                    current_pos[code] = 0.5 / len(valid_vol)
            else:
                # 全部权益/商品动量为负，配置债券+货币ETF（避险）
                current_pos = {c: 0.0 for c in etf_codes}
                current_pos['现金'] = 0.0
                # 配置国债和货币ETF
                bond_codes = [c for c in ['511260', '511990'] if c in etf_codes]
                for code in bond_codes:
                    current_pos[code] = 1.0 / len(bond_codes)

        # 记录持仓（转中文名）
        holding = {'date': date.strftime('%Y-%m-%d')}
        for c, v in current_pos.items():
            if c == '现金':
                holding['现金'] = round(v, 4)
            else:
                holding[GLOBAL_ETF_POOL[c]['name']] = round(v, 4)
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
    """保存策略净值JSON"""
    nav = calc_nav(returns)
    nav_norm = nav / nav.iloc[0]

    current_holding = holdings_log[-1] if holdings_log else {}

    data = {
        'strategy_name': '多元配置(风险平价)',
        'strategy_type': 'multi_asset',
        'metrics': metrics,
        'current_holding': current_holding,
        'etf_pool': {k: {'name': v['name'], 'region': v['region']} for k, v in GLOBAL_ETF_POOL.items()},
        'nav': [
            {'date': d.strftime('%Y-%m-%d'), 'nav': round(float(v), 4)}
            for d, v in nav_norm.items()
        ],
        'holdings': holdings_log[-52:],
    }

    path = os.path.join(STRATEGY_DIR, 'multi_asset.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  -> 已保存: {path}")
    return path


# ============================================================
# 主函数
# ============================================================
def main():
    os.makedirs(STRATEGY_DIR, exist_ok=True)

    # 1. 拉取数据
    etf_data = fetch_all_etf_data()

    if len(etf_data) < 5:
        print("\n错误: ETF数据不足!")
        sys.exit(1)

    # 2. 回测
    returns, positions, prices, holdings_log = backtest_multi_asset(etf_data)

    # 3. 计算指标
    metrics = calc_metrics(returns, '多元配置(风险平价)')
    metrics['strategy_type'] = 'multi_asset'
    metrics['description'] = '风险平价模型，12只全球ETF月频再平衡，动量过滤'

    # 4. 保存JSON
    save_strategy_json(returns, positions, holdings_log, metrics)

    # 5. 打印结果
    print("\n" + "=" * 60)
    print("多元配置策略回测结果")
    print("=" * 60)
    print(f"  策略: {metrics['name']}")
    print(f"  总收益: {metrics['total_return']}%")
    print(f"  年化: {metrics['annual_return']}%")
    print(f"  最大回撤: {metrics['max_drawdown']}%")
    print(f"  夏普: {metrics['sharpe']}")
    print(f"  Calmar: {metrics['calmar']}")
    print(f"  胜率: {metrics['win_rate']}%")
    print(f"  波动率: {metrics['volatility']}%")
    print(f"  区间: {metrics['start_date']} ~ {metrics['end_date']} ({metrics['trading_days']}交易日)")

    # 当前持仓
    if holdings_log:
        latest = holdings_log[-1]
        print(f"\n  最新持仓 ({latest['date']}):")
        for k, v in latest.items():
            if k != 'date' and v > 0:
                print(f"    {k}: {v*100:.1f}%")

    return metrics


if __name__ == '__main__':
    main()
