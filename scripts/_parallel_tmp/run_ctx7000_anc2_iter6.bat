@echo off
title ctx7000_anc2_iter6
cd /d "D:\workspace\hit\flowRAG -2.0"
call conda activate cogmait2
python "D:\workspace\hit\flowRAG -2.0\examples\quick_start.py" --batch --token-limit 7000 --anchor-nodes 2 --max-iterations 6 --workers 12 --auto-eval --eval-workers 12
echo.
echo ========================================
echo   ctx7000_anc2_iter6 完成
echo ========================================
