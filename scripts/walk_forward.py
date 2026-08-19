#!/usr/bin/env python3
"""
Walk-Forward参数稳定性检验
滚动窗口测试：用前N个月数据选参数，在后1个月验证
检验策略是否过拟合
"""

import sys, os, json, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from industry_rotation import INDUSTRY_ETF_POOL, fetch_all_etf_data, fetch_etf_data, calc_metrics
from multi_asset import GLOBAL_ETF_POOL, fetch_all_etf_data as fetch_global_data
warnings.filterwarnings('ignore')


def walk_forward_industry_rotation():
    """行业轮动walk-forward检验"""
    print("=" * 60)
    print("【Walk-Forward】行业轮动策略参数稳定性检验")
    print("=" * 60)

    etf_data = fetch_all_etf_data()
    close_dict = {}
    for code, df in etf_data.items():
        close_dict[code] = df.set_index('date')['close']
    prices = pd.DataFrame(close_dict).dropna(how='all').ffill().dropna()

    # walk-forward：每3个月为一个测试窗口
    # 用全部历史数据选参数（不滚动选参，而是检验不同参数在不同窗口的稳定性）
    test_windows = []
    dates = prices.index
    window_size = 60  # 60个交易日（约3个月）测试窗口

    for start in range(120, len(dates) - window_size, window_size):
        test_start = dates[start]
        test_end = dates[min(start + window_size - 1, len(dates) - 1)]
        test_windows.append((test_start, test_end))

    print(f"测试窗口数: {len(test_windows)}")
    print(f"每窗口: {window_size}交易日\n")

    # 测试参数组合
    param_sets = [
        {'ml': 20, 'ms': 10, 'k': 4},   # 当前最优
        {'ml': 20, 'ms': 20, 'k': 4},   # 变动短动量
        {'ml': 40, 'ms': 20, 'k': 4},   # 变动长动量
        {'ml': 20, 'ms': 10, 'k': 3},   # 变动K
        {'ml': 20, 'ms': 10, 'k': 2},   # 变动K
    ]

    results = []

    for params in param_sets:
        ml, ms, k = params['ml'], params['ms'], params['k']

        # 全量回测
        mom_long = prices / prices.shift(ml) - 1
        mom_short = prices / prices.shift(ms) - 1
        accel = mom_short - mom_long
        rs = prices / prices.rolling(20).mean() - 1
        combined = 0.4 * mom_long.rank(axis=1, ascending=False, pct=True) + 0.3 * accel.rank(axis=1, ascending=False, pct=True) + 0.3 * rs.rank(axis=1, ascending=False, pct=True)

        # 月频调仓
        rebal_dates = []
        last_month = None
        for d in prices.index:
            if d.month != last_month:
                rebal_dates.append(d)
                last_month = d.month
        if prices.index[-1] not in rebal_dates:
            rebal_dates.append(prices.index[-1])

        ec = prices.columns.tolist()
        dr = prices.pct_change().fillna(0)
        pos = pd.DataFrame(0.0, index=prices.index, columns=ec)
        cp = {c: 0.0 for c in ec}

        for i in range(len(rebal_dates) - 1):
            dt = rebal_dates[i]; nd = rebal_dates[i+1]
            if i >= 2:
                sr = combined.loc[dt]
                vs = sr[sr.notna()]
                if len(vs) >= 1:
                    tk = vs.nlargest(min(k, len(vs)))
                    cp = {c: 0.0 for c in ec}
                    for c in tk.index: cp[c] = 1.0 / len(tk)
            mask = (pos.index > dt) & (pos.index <= nd)
            for c in ec: pos.loc[mask, c] = cp.get(c, 0.0)

        sr_ret = pd.Series(0.0, index=prices.index)
        for col in ec: sr_ret += pos[col].shift(1) * dr[col]
        sr_ret = sr_ret.fillna(0)
        sr_ret -= pos.diff().abs().sum(axis=1) * 0.0005

        # 检查每个窗口的表现
        window_annual_returns = []
        for ws, we in test_windows:
            w_mask = (sr_ret.index >= ws) & (sr_ret.index <= we)
            w_ret = sr_ret[w_mask]
            if len(w_ret) > 10:
                w_annual = (1 + w_ret).prod() ** (252/len(w_ret)) - 1
                # 同期沪深300
                hs300_ret = prices['510300'].pct_change().fillna(0)[w_mask]
                hs300_annual = (1 + hs300_ret).prod() ** (252/len(hs300_ret)) - 1
                window_annual_returns.append((ws.strftime('%Y-%m'), w_annual, hs300_annual, w_annual - hs300_annual))

        # 统计
        excesses = [x[3] for x in window_annual_returns]
        win_rate = sum(1 for e in excesses if e > 0) / len(excesses) if excesses else 0
        avg_excess = np.mean(excesses) if excesses else 0
        std_excess = np.std(excesses) if excesses else 0

        full_metrics = calc_metrics(sr_ret, '')
        results.append({
            'params': f"ML={ml} MS={ms} K={k}",
            'annual': full_metrics['annual_return'],
            'sharpe': full_metrics['sharpe'],
            'max_dd': full_metrics['max_drawdown'],
            'wf_windows': len(window_annual_returns),
            'wf_win_rate': round(win_rate * 100, 1),
            'wf_avg_excess': round(avg_excess * 100, 1),
            'wf_std_excess': round(std_excess * 100, 1),
            'wf_details': window_annual_returns,
        })

    # 打印结果
    print(f"{'参数':16s} {'年化':>6} {'夏普':>6} {'回撤':>6} {'WF窗口':>6} {'WF胜率':>6} {'WF均超额':>8} {'WF超额std':>9}")
    for r in results:
        print(f"{r['params']:16s} {r['annual']:6.1f} {r['sharpe']:6.3f} {r['max_dd']:6.1f} {r['wf_windows']:6d} {r['wf_win_rate']:5.1f}% {r['wf_avg_excess']:7.1f}% {r['wf_std_excess']:8.1f}%")

    # 详细看最优参数的各窗口表现
    best = max(results, key=lambda x: x['wf_avg_excess'])
    print(f"\n最优参数({best['params']})各窗口超额:")
    for w, ar, bh, ex in best['wf_details']:
        print(f"  {w}: 策略{ar:.1%} vs 沪深300 {bh:.1%}, 超额{'+' if ex>=0 else ''}{ex:.1%}")

    return results


def walk_forward_multi_asset():
    """多元配置walk-forward检验"""
    print("\n" + "=" * 60)
    print("【Walk-Forward】多元配置策略参数稳定性检验")
    print("=" * 60)

    import multi_asset as ma
    etf_data = ma.fetch_all_etf_data()

    close_dict = {}
    for code, df in etf_data.items():
        if len(df) >= 200:
            close_dict[code] = df.set_index('date')['close']
    prices = pd.DataFrame(close_dict).dropna(how='all').ffill().dropna()
    dr = prices.pct_change().fillna(0)

    # 测试不同参数
    param_sets = [
        {'vol_window': 60, 'mom_window': 60, 'bond': 0.10, 'cash': 0.05},
        {'vol_window': 40, 'mom_window': 60, 'bond': 0.10, 'cash': 0.05},
        {'vol_window': 60, 'mom_window': 40, 'bond': 0.10, 'cash': 0.05},
        {'vol_window': 60, 'mom_window': 60, 'bond': 0.15, 'cash': 0.05},
        {'vol_window': 60, 'mom_window': 60, 'bond': 0.05, 'cash': 0.05},
    ]

    # walk-forward窗口
    test_windows = []
    dates = prices.index
    window_size = 60
    for start in range(120, len(dates) - window_size, window_size):
        test_windows.append((dates[start], dates[min(start + window_size - 1, len(dates) - 1)]))

    print(f"测试窗口数: {len(test_windows)}\n")

    results = []
    for params in param_sets:
        vw, mw = params['vol_window'], params['mom_window']
        bond_a, cash_a = params['bond'], params['cash']
        equity_a = 1.0 - bond_a - cash_a

        vol_df = dr.rolling(vw).std() * np.sqrt(252)
        mom_df = prices / prices.shift(mw) - 1

        rebal_dates = []
        last_month = None
        for d in prices.index:
            if d.month != last_month:
                rebal_dates.append(d)
                last_month = d.month
        if prices.index[-1] not in rebal_dates:
            rebal_dates.append(prices.index[-1])

        ec = prices.columns.tolist()
        BOND_CODES = ['511260', '511990']
        pos = pd.DataFrame(0.0, index=prices.index, columns=ec)
        cp = {c: 0.0 for c in ec}

        for i in range(len(rebal_dates) - 1):
            dt = rebal_dates[i]; nd = rebal_dates[i+1]
            if i >= 3:
                vol_row = vol_df.loc[dt]
                mom_row = mom_df.loc[dt]
                valid_mask = (mom_row > 0) & vol_row.notna() & (vol_row > 0)
                for c in BOND_CODES:
                    if c in valid_mask.index: valid_mask[c] = False
                valid_vol = vol_row[valid_mask]

                if len(valid_vol) >= 2:
                    ret_up = dr.loc[:dt]
                    cov = ret_up.iloc[-vw:].cov() * 252 if len(ret_up) > vw else None
                    w = ma.risk_parity_weights(vol_row[valid_mask], cov)
                    w = w[w > 0] * equity_a
                    cp = {c: 0.0 for c in ec}
                    for c in w.index: cp[c] = float(w[c])
                    if '511260' in ec: cp['511260'] = bond_a
                    if '511990' in ec: cp['511990'] = cash_a
                else:
                    cp = {c: 0.0 for c in ec}
                    if '511260' in ec: cp['511260'] = bond_a
                    if '511990' in ec: cp['511990'] = cash_a + equity_a
            mask = (pos.index > dt) & (pos.index <= nd)
            for c in ec: pos.loc[mask, c] = cp.get(c, 0.0)

        sr_ret = pd.Series(0.0, index=prices.index)
        for col in ec: sr_ret += pos[col].shift(1) * dr[col]
        sr_ret = sr_ret.fillna(0)
        sr_ret -= pos.diff().abs().sum(axis=1) * 0.0004

        # walk-forward各窗口
        window_returns = []
        for ws, we in test_windows:
            w_mask = (sr_ret.index >= ws) & (sr_ret.index <= we)
            w_ret = sr_ret[w_mask]
            if len(w_ret) > 10:
                w_annual = (1 + w_ret).prod() ** (252/len(w_ret)) - 1
                window_returns.append((ws.strftime('%Y-%m'), w_annual))

        annual_returns = [x[1] for x in window_returns]
        win_rate = sum(1 for a in annual_returns if a > 0) / len(annual_returns) if annual_returns else 0
        avg = np.mean(annual_returns) if annual_returns else 0
        std = np.std(annual_returns) if annual_returns else 0

        full_metrics = calc_metrics(sr_ret, '')
        results.append({
            'params': f"VW={vw} MW={mw} B={bond_a:.0%} C={cash_a:.0%}",
            'annual': full_metrics['annual_return'],
            'sharpe': full_metrics['sharpe'],
            'max_dd': full_metrics['max_drawdown'],
            'wf_windows': len(window_returns),
            'wf_win_rate': round(win_rate * 100, 1),
            'wf_avg_annual': round(avg * 100, 1),
            'wf_std': round(std * 100, 1),
            'wf_details': window_returns,
        })

    print(f"{'参数':24s} {'年化':>6} {'夏普':>6} {'回撤':>6} {'WF窗口':>6} {'WF胜率':>6} {'WF均年化':>8} {'WF std':>7}")
    for r in results:
        print(f"{r['params']:24s} {r['annual']:6.1f} {r['sharpe']:6.3f} {r['max_dd']:6.1f} {r['wf_windows']:6d} {r['wf_win_rate']:5.1f}% {r['wf_avg_annual']:7.1f}% {r['wf_std']:6.1f}%")

    best = max(results, key=lambda x: x['wf_avg_annual'])
    print(f"\n最优参数({best['params']})各窗口年化:")
    for w, ar in best['wf_details']:
        print(f"  {w}: {ar:.1%}")

    return results


if __name__ == '__main__':
    ir_results = walk_forward_industry_rotation()
    ma_results = walk_forward_multi_asset()

    # 保存报告
    report = {
        'test_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'industry_rotation': [{'params': r['params'], 'annual': r['annual'], 'sharpe': r['sharpe'],
                              'wf_win_rate': r['wf_win_rate'], 'wf_avg_excess': r['wf_avg_excess'],
                              'wf_std_excess': r['wf_std_excess']} for r in ir_results],
        'multi_asset': [{'params': r['params'], 'annual': r['annual'], 'sharpe': r['sharpe'],
                        'wf_win_rate': r['wf_win_rate'], 'wf_avg_annual': r['wf_avg_annual'],
                        'wf_std': r['wf_std']} for r in ma_results],
    }

    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'walk_forward_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {report_path}")
