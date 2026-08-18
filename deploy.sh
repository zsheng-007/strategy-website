#!/bin/bash
# 志胜投资策略网站 - GitHub部署脚本
# 用法: ./deploy.sh <GitHub用户名> <仓库名> <Personal Access Token>

set -e

USERNAME=$1
REPO=$2
TOKEN=$3

if [ -z "$USERNAME" ] || [ -z "$REPO" ] || [ -z "$TOKEN" ]; then
    echo "用法: ./deploy.sh <GitHub用户名> <仓库名> <PAT>"
    echo "示例: ./deploy.sh zhisheng strategy-website <TOKEN>"
    exit 1
fi

echo "=== 志胜投资策略网站部署 ==="
echo "用户: $USERNAME"
echo "仓库: $REPO"
echo ""

cd /workspace/strategy-website

# 初始化git
if [ ! -d .git ]; then
    git init
    git branch -M main
fi

# 添加远程仓库
git remote remove origin 2>/dev/null || true
git remote add origin "https://${USERNAME}:${TOKEN}@github.com/${USERNAME}/${REPO}.git"

# 提交代码
git add .
git commit -m "志胜投资策略网站 - 初始部署

- 3个ETF轮动策略回测（动量轮动/等权再平衡/相对强弱动态配比）
- 标的：价值ETF(512040) + 成长ETF(159259)
- 基准：中证A500(000922) + 中证500(000905)
- 周频调仓，日频数据更新
- GitHub Actions自动更新净值" || true

# 推送
git push -u origin main --force

echo ""
echo "=== 代码已推送！ ==="
echo ""
echo "下一步操作："
echo "1. 打开 https://github.com/${USERNAME}/${REPO}/settings/pages"
echo "   - Source: Deploy from a branch"
echo "   - Branch: main / (root)"
echo "   - 保存"
echo ""
echo "2. 打开 https://github.com/${USERNAME}/${REPO}/settings/actions"
echo "   - Workflow permissions: Read and write permissions"
echo "   - 保存"
echo ""
echo "3. 网站地址: https://${USERNAME}.github.io/${REPO}/"
echo ""
echo "4. 手动触发一次数据更新："
echo "   https://github.com/${USERNAME}/${REPO}/actions/workflows/daily_update.yml"
