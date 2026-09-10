#!/bin/bash
# ============================================================
# 志胜投资策略网站 - 一键部署脚本
# 在本地电脑（Mac/Linux/WSL）上运行
# 用法: ./deploy.sh
# ============================================================

# 已内置你的GitHub信息（如需修改请直接编辑）
USERNAME="zsheng-007"
REPO="strategy-website"
TOKEN="<TOKEN>"

echo "=== 志胜投资策略网站部署 ==="
echo "用户: $USERNAME"
echo "仓库: $REPO"
echo ""

# 切换到项目目录（脚本所在目录）
cd "$(dirname "$0")"

# 1. 创建GitHub仓库（通过API）
echo "[1/5] 创建GitHub仓库..."
RESP=$(curl -s -X POST "https://api.github.com/user/repos" \
  -H "Authorization: token $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -d "{\"name\":\"$REPO\",\"description\":\"志胜投资策略 - A股ETF量化策略研究与展示\",\"private\":false,\"has_pages\":true,\"auto_init\":false}")

if echo "$RESP" | grep -q "html_url"; then
  echo "  ✓ 仓库创建成功"
elif echo "$RESP" | grep -q "already"; then
  echo "  ℹ 仓库已存在，继续推送"
else
  echo "  ⚠ 创建仓库返回: $RESP"
  echo "  继续尝试推送..."
fi

# 2. 初始化git
echo "[2/5] 初始化Git仓库..."
if [ ! -d .git ]; then
  git init
  git branch -M main
fi
git config user.name "$USERNAME"
git config user.email "$USERNAME@users.noreply.github.com"

# 3. 添加远程仓库
echo "[3/5] 配置远程仓库..."
git remote remove origin 2>/dev/null || true
git remote add origin "https://${USERNAME}:${TOKEN}@github.com/${USERNAME}/${REPO}.git"

# 4. 提交代码
echo "[4/5] 提交代码..."
git add .
git commit -m "志胜投资策略网站 - 初始部署

- 3个ETF轮动策略回测（动量轮动/等权再平衡/相对强弱动态配比）
- 标的：价值ETF(512040) + 成长ETF(159259)
- 基准：中证A500(000922) + 中证500(000905)
- 周频调仓，日频数据自动更新
- GitHub Actions定时任务" 2>/dev/null || echo "  ℹ 无新改动需提交"

# 5. 推送
echo "[5/5] 推送代码到GitHub..."
git push -u origin main --force 2>&1

if [ $? -eq 0 ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║          ✓ 代码推送成功！                            ║"
  echo "╚══════════════════════════════════════════════════════╝"
  echo ""
  echo "接下来请完成以下3步（在浏览器中操作）："
  echo ""
  echo "━━ 步骤1：启用 GitHub Pages ━━"
  echo "  打开: https://github.com/$USERNAME/$REPO/settings/pages"
  echo "  - Source: Deploy from a branch"
  echo "  - Branch: main / (root)"
  echo "  - 点击 Save"
  echo ""
  echo "━━ 步骤2：配置 Actions 权限 ━━"
  echo "  打开: https://github.com/$USERNAME/$REPO/settings/actions"
  echo "  - Workflow permissions → 选择 Read and write permissions"
  echo "  - 勾选 Allow GitHub Actions to create and approve pull requests"
  echo "  - 点击 Save"
  echo ""
  echo "━━ 步骤3：手动触发首次数据更新 ━━"
  echo "  打开: https://github.com/$USERNAME/$REPO/actions/workflows/daily_update.yml"
  echo "  - 点击 'Run workflow' → Run workflow"
  echo ""
  echo "━━ 完成！网站地址 ━━"
  echo "  https://$USERNAME.github.io/$REPO/"
  echo "  （Pages部署约需1-2分钟生效）"
  echo ""
else
  echo ""
  echo "✗ 推送失败，请检查："
  echo "  1. Token是否有效（需repo + workflow权限）"
  echo "  2. 网络是否可访问github.com"
  echo "  3. 手动创建仓库: https://github.com/new"
  echo "     名称填: $REPO"
  echo "     然后重新运行本脚本"
fi
