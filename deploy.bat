@echo off
chcp 65001 >nul
REM ============================================================
REM 志胜投资策略网站 - Windows一键部署脚本
REM 需要已安装Git for Windows
REM ============================================================

set USERNAME=zsheng-007
set REPO=strategy-website
set TOKEN=<TOKEN>

echo === 志胜投资策略网站部署 ===
echo 用户: %USERNAME%
echo 仓库: %REPO%
echo.

REM 切换到脚本所在目录
cd /d "%~dp0"

echo [1/5] 创建GitHub仓库...
curl -s -X POST "https://api.github.com/user/repos" -H "Authorization: token %TOKEN%" -H "Accept: application/vnd.github+json" -d "{\"name\":\"%REPO%\",\"description\":\"志胜投资策略\",\"private\":false,\"has_pages\":true,\"auto_init\":false}"

echo.
echo [2/5] 初始化Git仓库...
if not exist .git (
    git init
    git branch -M main
)
git config user.name "%USERNAME%"
git config user.email "%USERNAME%@users.noreply.github.com"

echo [3/5] 配置远程仓库...
git remote remove origin 2>nul
git remote add origin "https://%USERNAME%:%TOKEN%@github.com/%USERNAME%/%REPO%.git"

echo [4/5] 提交代码...
git add .
git commit -m "志胜投资策略网站 - 初始部署"

echo [5/5] 推送代码到GitHub...
git push -u origin main --force

echo.
echo === 部署完成！ ===
echo.
echo 接下来请在浏览器中完成以下配置：
echo.
echo 1. 启用GitHub Pages:
echo    https://github.com/%USERNAME%/%REPO%/settings/pages
echo    Source: Deploy from a branch → Branch: main / (root) → Save
echo.
echo 2. 配置Actions权限:
echo    https://github.com/%USERNAME%/%REPO%/settings/actions
echo    Workflow permissions: Read and write permissions → Save
echo.
echo 3. 手动触发首次数据更新:
echo    https://github.com/%USERNAME%/%REPO%/actions/workflows/daily_update.yml
echo    点击 Run workflow
echo.
echo 网站地址: https://%USERNAME%.github.io/%REPO%/
echo.
pause
