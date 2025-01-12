#!/bin/bash

# ログファイルのパスを定義
LOGFILE="market_data_updater.log"

# market_data_updater.pyをバックグラウンドで実行し、出力をログにリダイレクト
nohup python market_data_updater.py > "$LOGFILE" 2>&1 &

# プロセスIDを表示
echo "Process started with PID: $!"

