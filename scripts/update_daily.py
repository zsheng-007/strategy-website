#!/usr/bin/env python3
"""
志胜投资策略 - 日频数据更新脚本
拉取最新行情 → 更新策略净值JSON → 更新基准JSON → 更新summary.json
检测调仓日推送消息通知
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
    get_rebalance_dates,
)
import industry_rotation
import industry_rotation_v2
import multi_asset

# 推送通知配置（通过环境变量传入，避免硬编码）
# 支持企业微信机器人 / 钉钉机器人 / 飞书机器人 / 自定义webhook
# 在GitHub Actions secrets中设置 NOTIFY_WEBHOOK 和 NOTIFY_SECRET
NOTIFY_WEBHOOK = os.environ.get('NOTIFY_WEBHOOK', '')  # webhook地址
NOTIFY_SECRET = os.environ.get('NOTIFY_SECRET', '')    # 签名密钥(钉钉用)


def send_notification(title, content):
    """发送调仓消息通知"""
    if not NOTIFY_WEBHOOK:
        print("\n  [通知] 未配置NOTIFY_WEBHOOK，跳过推送。")
        print(f"  [通知] 标题: {title}")
        print(f"  [通知] 内容:\n{content}")
        return False

    # 判断webhook类型
    if 'qyapi.weixin.qq.com' in NOTIFY_WEBHOOK:
        # 企业微信机器人
        data = {
            'msgtype': 'markdown',
            'markdown': {'content': f"### {title}\n\n{content}"}
        }
    elif 'oapi.dingtalk.com' in NOTIFY_WEBHOOK:
        # 钉钉机器人
        data = {
            'msgtype': 'markdown',
            'markdown': {'title': title, 'text': f"### {title}\n\n{content}"}
        }
    elif 'open.feishu.cn' in NOTIFY_WEBHOOK:
        # 飞书机器人
        data = {
            'msg_type': 'text',
            'content': {'text': f"{title}\n\n{content}"}
        }
    else:
        # 通用webhook
        data = {'title': title, 'content': content, 'text': f"{title}\n\n{content}"}

    try:
        r = requests.post(NOTIFY_WEBHOOK, json=data, timeout=10, headers={'Content-Type': 'application/json'})
        if r.status_code == 200:
            print(f"  [通知] ✓ 推送成功: {title}")
            return True
        else:
            print(f"  [通知] ✗ 推送失败: HTTP {r.status_code} {r.text[:100]}")
            return False
    except Exception as e:
        print(f"  [通知] ✗ 推送异常: {e}")
        return False


def check_rebalance_and_notify(prices, positions_dict, all_metrics):
    """检查今日是否为调仓日，若是则推送消息"""
    today = prices.index[-1]
    is_rebal_day = today.weekday() == REBALANCE_WEEKDAY

    # 构建消息内容
    lines = []
    lines.append(f"**日期**: {today.strftime('%Y-%m-%d')} ({'调仓日' if is_rebal_day else '非调仓日'})")
    lines.append("")
    lines.append("**各策略最新持仓**:")
    lines.append("")

    for name, stype, _ in [('动量轮动', 'momentum', None), ('等权再平衡', 'equal_weight', None), ('相对强弱动态配比', 'relative_strength', None), ('行业轮动(高频)', 'industry_rotation_v2', None), ('多元配置(风险平价)', 'multi_asset', None)]:
        positions = positions_dict.get(stype)
        if positions is not None and len(positions) > 0:
            latest_pos = positions.iloc[-1]
            # 只显示非零持仓，行业轮动的列名是代码需转换
            pos_items = []
            for c, v in latest_pos.items():
                if v > 0.01:
                    if stype == 'industry_rotation':
                        display_name = industry_rotation.INDUSTRY_ETF_POOL.get(c, {}).get('name', c)
                        if c == '现金': display_name = '现金'
                    elif stype == 'industry_rotation_v2':
                        display_name = industry_rotation_v2.INDUSTRY_ETF_POOL.get(c, {}).get('name', c)
                        if c == '现金': display_name = '现金'
                    elif stype == 'multi_asset':
                        display_name = multi_asset.GLOBAL_ETF_POOL.get(c, {}).get('name', c)
                        if c == '现金': display_name = '现金'
                    else:
                        display_name = c
                    pos_items.append(f"{display_name}: {v*100:.0f}%")
            pos_str = ' | '.join(pos_items) if pos_items else '空仓'

            # 找对应metrics
            m = next((x for x in all_metrics if x.get('strategy_type') == stype), {})
            ret = m.get('total_return', 0)

            lines.append(f"- **{name}**: {pos_str}")
            lines.append(f"  累计收益: {'+' if ret >= 0 else ''}{ret}%")

    lines.append("")
    lines.append(f"> 数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> [查看网站](https://zsheng-007.github.io/strategy-website/)")

    content = '\n'.join(lines)

    if is_rebal_day:
        title = f"📊 志胜策略调仓提醒 {today.strftime('%m-%d')}"
    else:
        title = f"📈 志胜策略日报 {today.strftime('%m-%d')}"

    return send_notification(title, content), is_rebal_day


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
    all_positions = {}

    for name, stype, func in strategies:
        print(f"  计算 {name}...")
        returns, positions = func(prices)
        nav = calc_nav(returns)
        metrics = calc_metrics(returns, name)
        metrics['strategy_type'] = stype
        all_metrics.append(metrics)
        all_positions[stype] = positions

        # 保存JSON
        jdata = strategy_to_json(nav, name, stype, metrics, positions)
        path = os.path.join(STRATEGY_DIR, f'{stype}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(jdata, f, ensure_ascii=False)
        print(f"    -> 已更新: {path}")

    # 3.5 行业轮动2号策略（高频三因子）
    print("\n  计算行业轮动2号(高频三因子)...")
    try:
        v2_etf_data = industry_rotation_v2.fetch_all_etf_data()
        if len(v2_etf_data) >= 5:
            v2_returns, v2_positions, v2_prices, v2_holdings = industry_rotation_v2.backtest_industry_rotation_v2(v2_etf_data)
            v2_metrics = industry_rotation_v2.calc_metrics(v2_returns, '行业轮动(高频三因子)')
            v2_metrics['strategy_type'] = 'industry_rotation_v2'
            all_metrics.append(v2_metrics)
            all_positions['industry_rotation_v2'] = v2_positions
            industry_rotation_v2.save_strategy_json(v2_returns, v2_positions, v2_holdings, v2_metrics)
            print(f"    -> 已更新: industry_rotation_v2.json")
    except Exception as e:
        print(f"    行业轮动2号更新失败: {e}")

    # 3.6 多元配置策略（跨境ETF，独立数据源）
    print("\n  计算多元配置(风险平价)...")
    try:
        ma_etf_data = multi_asset.fetch_all_etf_data()
        if len(ma_etf_data) >= 5:
            ma_returns, ma_positions, ma_prices, ma_holdings = multi_asset.backtest_multi_asset(ma_etf_data)
            ma_metrics = multi_asset.calc_metrics(ma_returns, '多元配置(风险平价)')
            ma_metrics['strategy_type'] = 'multi_asset'
            all_metrics.append(ma_metrics)
            all_positions['multi_asset'] = ma_positions
            multi_asset.save_strategy_json(ma_returns, ma_positions, ma_holdings, ma_metrics)
            print(f"    -> 已更新: multi_asset.json")
    except Exception as e:
        print(f"    多元配置更新失败: {e}")

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

    print(f"\n=== 更新完成 ===")
    print(f"更新时间: {summary['update_time']}")
    print(f"数据区间: {summary['data_range']['start']} ~ {summary['data_range']['end']}")

    # 6. 打印各策略最新净值
    print("\n各策略最新表现:")
    for m in all_metrics:
        print(f"  {m['name']}: 总收益{m['total_return']}% 年化{m['annual_return']}% 回撤{m['max_drawdown']}% 夏普{m['sharpe']}")

    # 7. 调仓检测与推送通知
    print("\n=== 检查调仓与推送通知 ===")
    check_rebalance_and_notify(prices, all_positions, all_metrics)


if __name__ == '__main__':
    main()
