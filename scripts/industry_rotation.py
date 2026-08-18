#!/usr/bin/env python3
"""
行业轮动策略 - 动量+趋势双因子
策略原型：改良版"均线能量"策略（年化31.3%，夏普1.32，2013-2025回测）
优化点：加入斜率×R²动量因子，双因子合成得分

ETF池：12只行业/宽基ETF覆盖周期/消费/科技/金融/医药
调仓频率：周频（每周五）
选基规则：双因子合成得分前2名且得分>0，否则空仓
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

REBALANCE_WEEKDAY = 4  # 周五调仓
MA_WINDOW = 20  # 均线窗口
MOMENTUM_WINDOW = 20  # 动量回看窗口
TOP_K = 2  # 持有前K名（参数优化最优）
TRANSACTION_COST = 0.0003  # 单边万分之三
MARKET_FILTER_WINDOW = 60  # 大盘择时窗口
MARKET_FILTER_THRESHOLD = -0.05  # 大盘60日收益<-5%才半仓

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
    """拉取全部行业ETF数据"""
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
def calc_ma_energy(prices_df, window=MA_WINDOW):
    """
    均线能量指标（改良版动量）
    衡量价格相对均线的累计偏离，判断趋势方向和强度
    正值=趋势向上，负值=趋势向下，绝对值越大趋势越强
    """
    close = prices_df['close'].astype(float)
    ma = close.rolling(window=window).mean()
    deviation = (close - ma) / ma

    # 均线能量 = 近window期偏离的累加
    energy = deviation.rolling(window=window).sum()
    # 确保index是日期
    energy.index = prices_df['date'].values
    return energy


def calc_slope_r2(prices_df, window=MOMENTUM_WINDOW):
    """
    斜率×R² 动量因子
    对数价格在过去N日做线性回归
    斜率反映趋势速度（年化），R²反映趋势确定性
    Score = 年化斜率 × R²
    """
    log_price = np.log(prices_df['close'].astype(float))

    scores = pd.Series(index=prices_df['date'].values, dtype=float)

    for i in range(window, len(log_price)):
        y = log_price.iloc[i-window:i].values
        x = np.arange(window)

        # 线性回归
        n = len(x)
        x_mean = x.mean()
        y_mean = y.mean()

        ss_xx = ((x - x_mean) ** 2).sum()
        ss_xy = ((x - x_mean) * (y - y_mean)).sum()
        ss_yy = ((y - y_mean) ** 2).sum()

        if ss_xx == 0 or ss_yy == 0:
            scores.iloc[i] = 0
            continue

        slope = ss_xy / ss_xx
        r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

        # 年化斜率（日频→年频：×252）
        annualized_slope = slope * 252

        scores.iloc[i] = annualized_slope * r_squared

    return scores


def calc_combined_score(etf_data):
    """
    计算所有ETF的双因子合成得分
    合成得分 = 0.5 × Rank(均线能量) + 0.5 × Rank(斜率×R²)
    """
    all_energy = {}
    all_slope = {}

    for code, df in etf_data.items():
        all_energy[code] = calc_ma_energy(df)
        all_slope[code] = calc_slope_r2(df)

    # 对齐到相同日期
    energy_df = pd.DataFrame({code: s for code, s in all_energy.items()})
    slope_df = pd.DataFrame({code: s for code, s in all_slope.items()})

    # 按日期对齐
    common_dates = energy_df.index.intersection(slope_df.index)
    energy_df = energy_df.loc[common_dates]
    slope_df = slope_df.loc[common_dates]

    # 计算截面排名（每期对所有ETF排名，1=最好）
    energy_rank = energy_df.rank(axis=1, ascending=False, pct=True)
    slope_rank = slope_df.rank(axis=1, ascending=False, pct=True)

    # 合成得分（排名越小越好 → 转换为得分越大越好）
    # 使用 1 - rank_pct 作为得分（rank_pct=0最差→得分1, rank_pct=1最好→得分0 → 反转）
    # 实际：rank ascending=False pct=True → 最好的pct接近1
    # 合成得分 = (energy_rank + slope_rank) / 2，值越大越好
    combined_score = (energy_rank + slope_rank) / 2

    # 同时保留原始因子值，用于判断正负
    return combined_score, energy_df, slope_df


# ============================================================
# 回测引擎
# ============================================================
def get_rebalance_dates(prices, weekday=REBALANCE_WEEKDAY):
    """获取调仓日"""
    dates = []
    for d in prices.index:
        if d.weekday() == weekday:
            dates.append(d)
    if prices.index[-1] not in dates:
        dates.append(prices.index[-1])
    return dates


def backtest_industry_rotation(etf_data):
    """
    行业轮动回测
    - 每周五计算双因子合成得分
    - 选前K名且得分>0的ETF等权持有
    - 全部得分≤0时空仓
    """
    print("\n运行行业轮动策略...")

    # 构建收盘价矩阵（用ETF代码作为列名，确保与因子一致）
    close_dict = {}
    for code, df in etf_data.items():
        close_dict[code] = df.set_index('date')['close']
    prices = pd.DataFrame(close_dict).dropna(how='all').ffill().dropna()

    # 计算因子得分
    combined_score, energy_df, slope_df = calc_combined_score(etf_data)

    # 对齐日期
    common_dates = prices.index.intersection(combined_score.index)
    prices = prices.loc[common_dates]
    combined_score = combined_score.loc[common_dates]
    energy_df = energy_df.loc[common_dates]
    slope_df = slope_df.loc[common_dates]

    # 获取调仓日
    rebal_dates = get_rebalance_dates(prices)

    etf_codes = prices.columns.tolist()
    positions = pd.DataFrame(0.0, index=prices.index, columns=etf_codes + ['现金'])
    daily_returns = prices.pct_change().fillna(0)

    current_pos = {c: 0.0 for c in etf_codes}
    current_pos['现金'] = 1.0  # 初始全仓现金

    holdings_log = []

    for i in range(len(rebal_dates) - 1):
        date = rebal_dates[i]
        next_date = rebal_dates[i + 1]

        if i >= 2:  # 需要足够数据计算因子
            # 获取当日得分
            score_row = combined_score.loc[date]
            energy_row = energy_df.loc[date]

            # 大盘择时：用沪深300(510300)过去N日收益判断
            if '510300' in prices.columns:
                hs300 = prices['510300']
                if len(hs300.loc[:date]) > MARKET_FILTER_WINDOW:
                    market_return = hs300.loc[date] / hs300.loc[:date].iloc[-MARKET_FILTER_WINDOW] - 1
                else:
                    market_return = 0
            else:
                market_return = 0

            # 筛选：得分非NaN
            valid_scores = score_row[score_row.notna()]

            if len(valid_scores) >= 1:
                # 选前K名
                top_k = valid_scores.nlargest(min(TOP_K, len(valid_scores)))
                # 大盘择时：弱市减半仓
                position_scale = 1.0 if market_return > MARKET_FILTER_THRESHOLD else 0.5

                current_pos = {c: 0.0 for c in etf_codes}
                current_pos['现金'] = 1.0 - position_scale
                for code in top_k.index:
                    current_pos[code] = position_scale * (1.0 / len(top_k))
            else:
                # 无有效数据，空仓
                current_pos = {c: 0.0 for c in etf_codes}
                current_pos['现金'] = 1.0

        # 记录持仓（转成中文名方便展示）
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

    # 计算策略收益（现金部分收益为0）
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

    # 当前持仓
    current_holding = holdings_log[-1] if holdings_log else {}

    data = {
        'strategy_name': '行业轮动(双因子)',
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

    # 1. 拉取数据
    etf_data = fetch_all_etf_data()

    if len(etf_data) < 5:
        print("\n错误: ETF数据不足!")
        sys.exit(1)

    # 2. 回测
    returns, positions, prices, holdings_log = backtest_industry_rotation(etf_data)

    # 3. 计算指标
    metrics = calc_metrics(returns, '行业轮动(双因子)')
    metrics['strategy_type'] = 'industry_rotation'
    metrics['description'] = '均线能量+斜率×R²双因子，12只行业ETF周频轮动，前2名持有'

    # 4. 保存JSON
    save_strategy_json(returns, positions, holdings_log, metrics)

    # 5. 打印结果
    print("\n" + "=" * 60)
    print("行业轮动策略回测结果")
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
