#!/bin/bash
# ============================================================
# 志胜投资策略网站 - 一键部署脚本
# 在本地电脑（Mac/Linux/WSL）上运行
# 用法: ./deploy.sh
# 提示: 请先 export GITHUB_TOKEN=xxx 再执行本脚本
# ====================================

USERNAME="${GITHUB_USERNAME:-zsheng-007}"
REPO="strategy-website"
TOKEN="${GITHUB_TOKEN:?请先 export GITHUB_TOKEN=your_token}"

echo "=== 志胜投资策略网站部署 ==="
echo "用户: $USERNAME"
echo "仓库: $REPO"
echo ""

# 切换到项目目录（脚本所在目录）
cd "$(dirname "$0")"

git add -A
git commit -m "📊 数据更新 $(date '+%Y-%m-%d %H:%M')" || echo "无变更"
git push https://${USERNAME}:${TOKEN}@github.com/${USERNAME}/${REPO}.git main
