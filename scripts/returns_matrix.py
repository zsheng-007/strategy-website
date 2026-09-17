#!/usr/bin/env python3
"""
多窗口收益统计矩阵生成器
=======================
为6个策略生成 6个时间窗口（近一日/周/月/季/年/三年）的收益统计，
区分两套口径：

  1. 实际（近期实盘）：以真实运行区间为基准，计算各窗口的实际收益与实际年化。
     - 窗口超出真实数据长度时记为 None（前端显示"数据不足"）
     - 年化用几何年化：(1+总收益)^(252/交易日数) - 1

  2. 回测（全样本）：策略在全可用历史（2016年至今，视ETF上市日期而定）上的
     完整回测收益与年化，作为策略设计能力的参考基准。

输出：data/returns_matrix.json
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
STRATEGY_DIR = os.path.join(DATA_DIR, 'strategies')

STRATEGY_KEYS = [
    'momentum', 'equal_weight', 'relative_strength',
    'industry_rotation_v2', 'no_industry', 'multi_asset',
]

# 窗口定义：(键, 显示名, 交易日数, 自然日数)
WINDOWS = [
    ('d1',   '近一日', 1,    1),
    ('w1',   '近一周', 5,    7),
    ('m1',   '近一月', 21,   30),
    ('q1',   '近一季', 63,   91),
    ('y1',   '近一年', 252,  365),
    ('y3',   '近三年', 756,  1095),
]


def load_nav(strategy_key):
    """加载策略净值序列"""
    path = os.path.join(STRATEGY_DIR, f'{strategy_key}.json')
    if not os.path.exists(path):
        return None, None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    nav = pd.Series(
        [r['nav'] for r in data['nav']],
        index=pd.to_datetime([r['date'] for r in data['nav']]),
    )
    return nav, data


def geometric_annual(total_return, trading_days):
    """几何年化：由总收益与交易日数换算年化"""
    years = trading_days / 252.0
    if years <= 0:
        return None
    # 极端亏损保护（总收益 <= -100% 时无意义）
    if total_return <= -1:
        return -1.0
    return (1 + total_return) ** (1 / years) - 1


def window_return(nav, trading_days):
    """取窗口收益：窗口末尾为最新净值，起点为倒数第 trading_days+1 个交易日

    返回 (总收益, 实际交易日数, 起始日, 结束日)，数据不足返回 (None, ...)
    """
    n = len(nav)
    if n < 2:
        return None, 0, None, None
    # 窗口起始位置（含端点，故 need = trading_days + 1）
    need = trading_days + 1
    if n < need:
        # 数据不足该窗口，但若窗口为1日且至少有2个点也能算
        if trading_days == 1 and n >= 2:
            start_idx = n - 2
        else:
            return None, 0, None, None
    else:
        start_idx = n - need

    seg = nav.iloc[start_idx:]
    actual_days = len(seg) - 1
    if actual_days <= 0:
        return None, 0, None, None
    total = (seg.iloc[-1] / seg.iloc[0]) - 1
    return total, actual_days, seg.index[0], seg.index[-1]


def build_matrix():
    print("=" * 70)
    print("生成多窗口收益统计矩阵")
    print("=" * 70)

    out = {
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'windows': [{'key': k, 'name': n} for k, n, _, _ in WINDOWS],
        'strategies': [],
    }

    for key in STRATEGY_KEYS:
        nav, raw = load_nav(key)
        if nav is None or len(nav) < 2:
            print(f"  ✗ {key}: 数据缺失")
            continue

        meta = raw.get('metrics', {})
        name = meta.get('name', key)
        bt = raw.get('backtest_full', {})

        # ── 口径1：全样本回测（策略在整个可用历史区间的表现） ──
        full_days = len(nav) - 1
        full_total = (nav.iloc[-1] / nav.iloc[0]) - 1
        full_annual = geometric_annual(full_total, full_days)

        # 若JSON里携带更长的 backtest_full（如未来接入10年独立回测），优先采用
        backtest = {
            'total_return': round(bt.get('total_return', full_total * 100), 2),
            'annual_return': round(bt.get('annual_return', (full_annual or 0) * 100), 2),
            'start_date': bt.get('start_date', nav.index[0].strftime('%Y-%m-%d')),
            'end_date': bt.get('end_date', nav.index[-1].strftime('%Y-%m-%d')),
            'trading_days': bt.get('trading_days', full_days),
            'max_drawdown': bt.get('max_drawdown', meta.get('max_drawdown')),
            'sharpe': bt.get('sharpe', meta.get('sharpe')),
        }

        # ── 口径2：近期实盘（各窗口实际收益 + 实际年化） ──
        windows = {}
        for wkey, wname, wdays, _ in WINDOWS:
            total, actual_days, sd, ed = window_return(nav, wdays)
            if total is None:
                windows[wkey] = {
                    'total_return': None, 'annual_return': None,
                    'trading_days': 0, 'sufficient': False,
                }
                continue
            annual = geometric_annual(total, actual_days)
            # 短窗口（<20个交易日）年化无实际意义：1日收益折算年化会被极度放大
            # （如+2.23% → 25948%），仅当窗口足够长时才提供年化
            if actual_days < 20:
                annual = None
            windows[wkey] = {
                'total_return': round(total * 100, 2),
                'annual_return': round(annual * 100, 2) if annual is not None else None,
                'trading_days': actual_days,
                'start_date': sd.strftime('%Y-%m-%d'),
                'end_date': ed.strftime('%Y-%m-%d'),
                'sufficient': True,
            }

        out['strategies'].append({
            'strategy_type': key,
            'name': name,
            'backtest': backtest,
            'windows': windows,
        })

        avail = [w for w in windows.values() if w['sufficient']]
        print(f"  ✓ {name:<24} 全样本({full_days}日/{full_days/252:.1f}年) "
              f"回测年化{backtest['annual_return']:>7.2f}%  可用窗口{len(avail)}/6")

    path = os.path.join(DATA_DIR, 'returns_matrix.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  -> 已保存: {path}")
    return out


if __name__ == '__main__':
    build_matrix()
