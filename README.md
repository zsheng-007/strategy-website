# 志胜投资策略网站

个人投资策略研究展示平台，聚焦A股ETF量化策略。

## 当前策略

基于**价值ETF(512040)**与**成长ETF(159259)**的风格轮动策略体系：

| 策略 | 逻辑 | 调仓频率 |
|---|---|---|
| 动量轮动 | 过去4周动量最强的ETF满仓持有 | 周频（每周五） |
| 等权再平衡 | 50/50等权配置，定期再平衡 | 周频（每周五） |
| 相对强弱动态配比 | 按动量强弱动态分配权重(20%-80%) | 周频（每周五） |

### 对比基准
- **中证A500指数(000922)** — A股核心资产整体表现
- **中证500指数(000905)** — 偏股混合基金指数885001为Wind独家编制，免费数据源无法获取，以此替代

## 数据更新

- **频率**：日频（每个交易日收盘后15:30 CST自动更新）
- **方式**：GitHub Actions定时任务自动拉取行情、更新净值、提交到仓库
- **数据源**：腾讯行情API

## 项目结构

```
strategy-website/
├── index.html                  # 网站主页面
├── data/
│   ├── summary.json            # 策略汇总信息
│   ├── strategies/
│   │   ├── momentum.json       # 动量轮动净值数据
│   │   ├── equal_weight.json   # 等权再平衡净值数据
│   │   └── relative_strength.json  # 相对强弱策略净值数据
│   └── benchmarks/
│       ├── a500.json           # 中证A500净值
│       └── csi500.json         # 中证500净值
├── scripts/
│   ├── backtest.py             # 回测引擎
│   └── update_daily.py         # 日频更新脚本
├── .github/workflows/
│   └── daily_update.yml        # GitHub Actions定时任务
└── README.md
```

## 部署到GitHub Pages

### 1. 创建仓库并推送代码

```bash
git init
git add .
git commit -m "志胜投资策略网站初始化"
git branch -M main
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

### 2. 启用GitHub Pages

1. 进入仓库 Settings → Pages
2. Source 选择 `Deploy from a branch`
3. Branch 选择 `main`，目录选 `/ (root)`
4. 保存后等待几分钟，网站将可通过 `https://<用户名>.github.io/<仓库名>/` 访问

### 3. 配置GitHub Actions权限

1. 进入仓库 Settings → Actions → General
2. Workflow permissions 选择 `Read and write permissions`
3. 勾选 `Allow GitHub Actions to create and approve pull requests`
4. 保存

### 4. 验证定时任务

- 进入仓库 Actions 页面
- 可手动触发 `日频数据更新` workflow 测试
- 每个交易日15:30 CST将自动执行

## 添加新策略

1. 在 `scripts/backtest.py` 中实现新策略函数
2. 在 `STRATEGY_META`（index.html）中添加策略元信息
3. 运行 `python3 scripts/backtest.py` 生成JSON数据
4. 提交代码，GitHub Pages自动部署

## 技术栈

- **回测引擎**：Python + pandas + numpy
- **数据源**：腾讯行情API（`web.ifzq.gtimg.cn`）
- **前端**：原生HTML + ECharts 5 + CSS变量
- **部署**：GitHub Pages + GitHub Actions

## 风险提示

本站内容仅供个人研究参考，不构成投资建议。过往业绩不代表未来表现。投资有风险，入市需谨慎。
