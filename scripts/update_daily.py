#!/usr/bin/env python3
"""
志胜投资策略 - 日频数据更新脚本
拉取最新行情 → 更新策略净值JSON → 更新基准JSON → 更新summary.json
用于GitHub Actions定时任务，每个交易日收盘后自动运行
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')

# 导入回测模块的配置和函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import (
    ETF_CONFIG, BENCHMARK_CONFIG, REBALANCE_WEEKDAY, MOMENTUM_LOOKBACK,
    TRANSACTION_COST, HEADERS, STRATEGY_DIR, BENCHMARK_DIR, OUTPUT_DIR,
    fetch_kline_tencent, prepare_prices, prepare_benchmarks,
    strategy_momentum_rotation, strategy_equal_weight, strategy_relative_strength,
    calc_nav, calc_metrics, strategy_to_json, benchmark_to_json,
)


def main():
    print(f"[{datetime.now()}] 开始日频数据更新...")

    # 1. 拉取最新数据
    print("\n=== 拉取最新行情数据 ===")
    etf_data = {}
    for code, info in ETF_CONFIG.items():
        print(f"  拉取 {info['fullname']}({code})...")
        df = fetch_kline_tencent(code, info['market'])
        if df is not None and len(df) > 0:
            etf_data[code] = df
            print(f"    -> {len(df)}条, 最新: {df['date'].iloc[-1].date()}")

    benchmark_data = {}
    for code, info in BENCHMARK_CONFIG.items():
        print(f"  拉取 {info['name']}({code})...")
        df = fetch_kline_tencent(code, info['market'], adjust='')
        if df is not None and len(df) > 0:
            benchmark_data[code] = df
            print(f"    -> {len(df)}条, 最新: {df['date'].iloc[-1].date()}")

    if len(etf_data) < 2:
        print("\n错误: ETF数据不足!")
        sys.exit(1)

    # 2. 数据处理
    prices = prepare_prices(etf_data)
    bench = prepare_benchmarks(benchmark_data) if benchmark_data else None

    print(f"\n对齐后交易日: {len(prices)}")
    print(f"日期范围: {prices.index[0].date()} ~ {prices.index[-1].date()}")

    # 3. 重新运行策略回测（全量计算，确保一致性）
    print("\n=== 更新策略净值 ===")
    strategies = [
        ('动量轮动', 'momentum', strategy_momentum_rotation),
        ('等权再平衡', 'equal_weight', strategy_equal_weight),
        ('相对强弱动态配比', 'relative_strength', strategy_relative_strength),
    ]

    all_metrics = []
    for name, stype, func in strategies:
        print(f"  计算 {name}...")
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
        print(f"    -> 已更新: {path}")

    # 4. 更新基准
    print("\n=== 更新基准数据 ===")
    if bench is not None:
        for code, info in BENCHMARK_CONFIG.items():
            name = info['name']
            if name in bench.columns:
                jdata = benchmark_to_json(bench[name], name, code)
                path = os.path.join(BENCHMARK_DIR, f"{info['file']}.json")
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(jdata, f, ensure_ascii=False)
                print(f"  -> {path}")

    # 5. 更新summary
    summary = {
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'etfs': {k: {kk: vv for kk, vv in v.items()} for k, v in ETF_CONFIG.items()},
        'benchmarks': {k: {kk: vv for kk, vv in v.items()} for k, v in BENCHMARK_CONFIG.items()},
        'benchmark_note': '偏股混合基金指数885001为Wind独家编制，免费数据源无法获取，以中证500指数(000905)替代',
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

    print(f"\n=== 更新完成 ===")
    print(f"更新时间: {summary['update_time']}")
    print(f"数据区间: {summary['data_range']['start']} ~ {summary['data_range']['end']}")

    # 6. 打印各策略最新净值
    print("\n各策略最新表现:")
    for m in all_metrics:
        print(f"  {m['name']}: 总收益{m['total_return']}% 年化{m['annual_return']}% 回撤{m['max_drawdown']}% 夏普{m['sharpe']}")


if __name__ == '__main__':
    main()
