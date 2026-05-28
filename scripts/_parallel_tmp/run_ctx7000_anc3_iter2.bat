@echo off
title ctx7000_anc3_iter2
cd /d "D:\workspace\hit\flowRAG"
call conda activate cogmait312
python "D:\workspace\hit\flowRAG\examples\quick_start.py" --batch --token-limit 7000 --anchor-nodes 3 --max-iterations 2 --workers 16 --auto-eval --eval-workers 32
echo.
echo ========================================
echo   ctx7000_anc3_iter2 完成
echo ========================================
pause
