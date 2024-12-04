# 列名定義
COLUMN_START_AT = 'start_at'
COLUMN_HIGH = 'high'
COLUMN_LOW = 'low'
COLUMN_CLOSE = 'close'
COLUMN_VOLUME = 'volume'
COLUMN_TURNOVER = 'turnover'
COLUMN_OPEN = 'open'
COLUMN_RSI = 'rsi'

COLUMN_UPPER_BAND = 'upper'
COLUMN_LOWER_BAND = 'lower'
COLUMN_MIDDLE_BAND = 'middle'
COLUMN_LOWER_BAND1 = 'lower1'
COLUMN_LOWER_BAND2 = 'lower2'
COLUMN_LOWER_BAND3 = 'lower3'
COLUMN_UPPER_BAND1 = 'upper1'
COLUMN_UPPER_BAND2 = 'upper2'
COLUMN_UPPER_BAND3 = 'upper3'
COLUMN_MIDDLE_BAND = 'middle'


COLUMN_EMA = 'ema'
COLUMN_SMA = 'sma'


COLUMN_MACD = 'macd'
COLUMN_MACDSIGNAL = 'macdsignal'
COLUMN_MACDHIST = 'macdhist'
COLUMN_P_DI = 'p_di'
COLUMN_M_DI = 'm_di'
COLUMN_ADX = 'adx'
COLUMN_ADXR = 'adxr'
COLUMN_BBVI = 'bbvi'
COLUMN_VOLUME_MA = 'volume_ma'
COLUMN_VOLUME_MA_DIFF = 'volume_ma_diff'
COLUMN_UPPER_DIFF = 'upper_diff'
COLUMN_LOWER_DIFF = 'lower_diff'
COLUMN_MIDDLE_DIFF = 'middle_diff'
COLUMN_EMA_DIFF = 'ema_diff'
COLUMN_SMA_DIFF = 'sma_diff'
COLUMN_RSI_SELL = 'rsi_sell'
COLUMN_RSI_BUY = 'rsi_buy'
COLUMN_DMI_DIFF = 'dmi_diff'
COLUMN_MACD_DIFF = 'macd_diff'
COLUMN_BOL_DIFF = 'bol_diff'
COLUMN_BAND_DIFF = 'band_diff'
COLUMN_DI_DIFF = 'di_diff'
COLUMN_ENTRY_DIFF = 'entry_diff'

# 列名定義
COLUMN_STATE = 'state'
COLUMN_BB_DIRECTION = 'bb_direction'
COLUMN_ENTRY_PRICE = 'entry_price'
COLUMN_EXIT_PRICE = 'exit_price'
COLUMN_CURRENT_PROFIT = 'current_profit'
COLUMN_BB_PROFIT = 'bb_profit'
COLUMN_PROFIT_MAX = 'profit_max'
COLUMN_PROFIT_MIN = 'profit_min'
COLUMN_PROFIT_MEAN = 'profit_mean'
COLUMN_PREDICTION = 'prediction'
COLUMN_PROFIT_MA = 'profit_ma'
COLUMN_SMA = 'sma'
COLUMN_EMA = 'ema'
COLUMN_ENTRY_TYPE = 'entry_type'
COLUMN_PANDL = 'pandl'
COLUMN_MAX_PANDL = 'max_pandl'
COLUMN_MIN_PANDL = 'min_pandl'
COLUMN_EXIT_REASON = 'exit_reason'
COLUMN_WCLPRICE = 'wclprice'
COLUMN_ADLINE = 'adline'
COLUMN_MFI = 'mfi'
COLUMN_ROC = 'roc'
COLUMN_ATR = 'atr'
COLUMN_EXIT_TIME = 'exit_time'
COLUMN_PRED_V1 = 'pred_v1'
COLUMN_PRED_V2 = 'pred_v2'
COLUMN_PRED_V3 = 'pred_v3'
COLUMN_PRED_V4 = 'pred_v4'
COLUMN_PRED_V5 = 'pred_v5'
COLUMN_PRED_V6 = 'pred_v6'
COLUMN_PRED_V7 = 'pred_v7'
COLUMN_PRED_V8 = 'pred_v8'
COLUMN_PRED_V9 = 'pred_v9'
COLUMN_PRED_V10 = 'pred_v10'
COLUMN_PRED_V11 = 'pred_v11'
COLUMN_PRED_V12 = 'pred_v12'
COLUMN_PRED_V13 = 'pred_v13'
COLUMN_PRED_V14 = 'pred_v14'
COLUMN_PRED_V15 = 'pred_v15'
COLUMN_PRED_V16 = 'pred_v16'
COLUMN_PRED_V17 = 'pred_v17'
COLUMN_PRED_V18 = 'pred_v18'
COLUMN_PRED_TARGET = 'pred_target'
COLUMN_ORDER_ID = 'order_id'

EXIT_REASON_LOSSCUT = 'losscut'
EXIT_REASON_NORMAL = 'normal'

# 状態名定義
STATE_IDLE = 'IdleState'
STATE_POSITION = 'PositionState'
STATE_ENTRY_PREPARATION = 'EntryPreparationState'
STATE_EXIT_PREPARATION = 'ExitPreparationState'

# イベント名定義
EVENT_EXIT_PREPARATION =  'ExitPreparationState'
EVENT_ENTER_PREPARATION = 'EntryPreparationState'
EVENT_POSITION = 'PositionState'
EVENT_IDLE = 'IdleState'

# 注意: プログラムから直接参照されるその他の状態名もリストアップする必要がありますが、
# 提供されたコード内では明示されていません。

# エラーメッセージ
ERROR_MESSAGE_BB_DIRECTION = 'error:{bb_direction}'

# 利益分析方向
BB_DIRECTION_UPPER = 'upper'
BB_DIRECTION_LOWER = 'lower'


# データフレームの新しいカラム名
COLUMN_DATE = 'date'  # 日付を保持するカラム名

# ログメッセージ
LOG_MESSAGE_SYSTEM_ERROR = 'error:{self.get_bb_direction()}'
LOG_MESSAGE_TRANSACTION = 'go to {state}: {bb_direction}'
LOG_MESSAGE_STATE_AND_TRANSITION = 'State and Transition Record'
LOG_TRANSITION_TO_ENTRY_PREPARATION = "ボリンジャーバンドに触れた。エントリー準備状態に遷移します。"
LOG_TRANSITION_TO_POSITION = "エントリーイベントが発生。ポジション状態に遷移します。"
LOG_TRANSITION_TO_IDLE_FROM_POSITION = "アイドル状態に遷移します。"
LOG_INVALID_EVENT = "エラー: 現在の状態では '{event}' イベントは無効です。"
# 追加の定数定義例

# ロギングメッセージ
LOG_MESSAGE_ENTER_PREPARATION = "ボリンジャーバンドに触れた。エントリー準備状態に遷移します。"
LOG_MESSAGE_ENTER_POSITION = "エントリーイベントが発生。ポジション状態に遷移します。"
LOG_MESSAGE_EXIT_POSITION = "アイドル状態に遷移します。"
LOG_MESSAGE_INVALID_EVENT = "エラー: 現在の状態では '{event}' イベントは無効です。"
LOG_MESSAGE_PREDICTION = 'prediction:{prediction}'
LOG_MESSAGE_LOSSCUT_CHECK = 'is_losscut:{is_losscut},current_profit:{current_profit}'


# ファイル名
FILENAME_RESULT_CSV = 'result.csv'

# その他プログラム内で使用される特定の数値や文字列
BB_PROFIT_THRESHOLD = 70  # RSIの売買判断基準などで使用される閾値

# 注意: さらに詳細な分析が必要な場合は、プログラム内で具体的に使用されているその他のハードコードされた値や
# 条件判断のための文字列リテラルも同様にリストアップする必要があります。

# トレードの勝敗を記録するカウンタのキー
# カウンタキーの定義
COUNTER_UPPER_1_WIN = 'upper_1_win'
COUNTER_UPPER_1_LOSE = 'upper_1_lose'
COUNTER_UPPER_0_WIN = 'upper_0_win'
COUNTER_UPPER_0_LOSE = 'upper_0_lose'
COUNTER_LOWER_1_WIN = 'lower_1_win'
COUNTER_LOWER_1_LOSE = 'lower_1_lose'
COUNTER_LOWER_0_WIN = 'lower_0_win'
COUNTER_LOWER_0_LOSE = 'lower_0_lose'

# エントリータイプ
ENTRY_TYPE_LONG = 'LONG'
ENTRY_TYPE_SHORT = 'SHORT'
PRED_TYPE_LONG = 1
PRED_TYPE_SHORT = 0


COUNTER_UPPER_1_LOSS = 'upper_1_loss'
COUNTER_UPPER_1_SAFE = 'upper_1_safe'
COUNTER_UPPER_0_LOSS = 'upper_0_loss'
COUNTER_UPPER_0_SAFE = 'upper_0_safe'
COUNTER_LOWER_1_LOSS = 'lower_1_loss'
COUNTER_LOWER_1_SAFE = 'lower_1_safe'
COUNTER_LOWER_0_LOSS = 'lower_0_loss'
COUNTER_LOWER_0_SAFE = 'lower_0_safe'

TRADING_EXIT_REASON_LOSSCUT = 'exit reson exceed losscut'


TIME_SERIES_PERIOD = 32
WAIT_FOR_ENTRY_COUNT = 1
ENTRY_COUNTER_THRESHOLD = 1
#PREDICTION_DISTANCE = 8
#PREDICTION_DISTANCE = 4
#PREDICTION_DISTANCE = 2



#bybit api
MAX_TRYOUT_HTTP_REQUEST = 5
MAX_TRYOUT_HTTP_REQUEST_SLEEP = 30
INITIAL_LEVERAGE = 1

#monodb collection name
AIML_TRACING = "aiml_tracing"
ROLLING_AI_DATA = "rolling_ai_data"
MARKET_DATA = "market_data"
MARKET_DATA_TECH = "market_data_tech"
MARKET_DATA_ML_UPPER = "market_data_ml_UPPER"
MARKET_DATA_ML_LOWER = "market_data_ml_LOWER"
MARKET_DATA_ML = "market_data_ml"
TRADING_LOG = "trading_log"
TRANSACTION_DATA = "transaction_data"
ACCOUNT_DATA = "account_data"
TRADING_LOG="trading_log"
TRADE_CONFIG="trade_config"

#config file name
CONFIG_FILENAME = 'aitrading_settings_ver2.json'
AIMODLE_CONFIG_FILENAME = 'aimodel_setting.json'

