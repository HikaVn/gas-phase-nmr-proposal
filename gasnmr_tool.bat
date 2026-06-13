@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title 気相NMR 精度設計ツール (GUI)

echo ===============================================
echo   気相NMR 精度設計ツール  GUI (Windows)
echo ===============================================
echo.

set "RAW=https://raw.githubusercontent.com/HikaVn/gas-phase-nmr-proposal/main"

REM --- Python を探す (py ランチャー優先) ---
set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
  echo [エラー] Python が見つかりませんでした。
  echo   https://www.python.org/downloads/ からインストールし、
  echo   インストーラの "Add Python to PATH" に必ずチェックを入れてください。
  echo.
  pause
  exit /b 1
)

REM --- スクリプトが無ければ GitHub から取得 ---
if not exist "bayes_comb_reanalysis.py" (
  echo bayes_comb_reanalysis.py をダウンロード中...
  call :download bayes_comb_reanalysis.py
)
if not exist "gasnmr_tool.py" (
  echo gasnmr_tool.py をダウンロード中...
  call :download gasnmr_tool.py
)
if not exist "gasnmr_tool.py" (
  echo [エラー] ダウンロードに失敗しました。ネット接続を確認してください。
  pause
  exit /b 1
)

REM --- 依存ライブラリを用意 (tkinter は python.org 版に同梱) ---
echo 依存ライブラリ (numpy, scipy, matplotlib) を確認中...
%PY% -m pip install --quiet --disable-pip-version-check numpy scipy matplotlib
if errorlevel 1 (
  echo [エラー] ライブラリのインストールに失敗しました。
  pause
  exit /b 1
)

echo.
echo GUI を起動します...
%PY% gasnmr_tool.py
if errorlevel 1 (
  echo.
  echo [注意] GUI の起動に失敗した可能性があります。
  echo   tkinter が無い場合は、バッチ実行を試してください:
  echo     %PY% gasnmr_tool.py --batch config.json
  pause
  exit /b 1
)
exit /b 0

:download
where curl >nul 2>nul
if %errorlevel%==0 (
  curl -fsSL -o "%~1" "%RAW%/%~1"
) else (
  powershell -NoProfile -Command "try { Invoke-WebRequest -Uri '%RAW%/%~1' -OutFile '%~1' } catch { exit 1 }"
)
exit /b 0
