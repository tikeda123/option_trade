import psycopg2
from psycopg2 import extras
import pandas as pd
import numpy as np
import decimal
import os, sys


# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.data_loader import DataLoader
from common.config_manager import ConfigManager
from common.constants import *

def dtype_mapping(dtype, column_name):
    """
    Pandasデータ型とカラム名をPostgreSQLのデータ型にマッピングする。
    特に、'start_at' カラムは 'TIMESTAMP' 型として扱う。
    """
    if column_name == 'start_at':
        return 'TIMESTAMP'
    elif column_name == 'date':
        return 'TEXT'
    elif dtype.startswith('int'):
        return 'INTEGER'
    elif dtype.startswith('float'):
        return 'FLOAT'
    elif dtype.startswith('datetime'):
        return 'TIMESTAMP'
    elif dtype.startswith('object'):
        return 'FLOAT'
    else:
        return 'TEXT'


class DataLoaderDB(DataLoader):
    """
    データベースからデータをロードするためのDataLoaderのサブクラスです。

    Attributes:
        config_manager (ConfigManager): 設定ファイルを管理するオブジェクト。
        table_name (str): データをロードするデータベーステーブルの名前。
        conn (psycopg2.connection): データベースへの接続。

    Args:
        config_fullpath (str): 設定ファイルのフルパス。
        table_name (Optional[str]): データをロードするテーブルの名前。指定しない場合は、設定ファイルから生成されます。
    """

    def __init__(self,table_name=None):
        """
        インスタンスを初期化し、データベース接続を確立します。
        """
        super().__init__()

        self.config_manager = ConfigManager()
        db_conf = self.config_manager.get('DATABASE')
        self.table_name = table_name or self.make_table_name()

        self.conn = psycopg2.connect(
            dbname=db_conf['DBNAME'],
            user=db_conf['USER'],
            password=db_conf['PASSWORD'],
            host=db_conf['HOST']
        )

    def __del__(self):
        """
        インスタンスの削除時にデータベース接続を閉じます。
        """
        self.conn.close()

    def set_table_name(self,table_name):
        self.table_name = table_name

    def make_table_name(self,table_name=None)->str:
        """
        デフォルトのテーブル名を生成または設定します。

        Args:
            table_name (Optional[str]): テーブル名。指定しない場合は、設定ファイルからシンボルとインターバルを使用して生成されます。

        Returns:
            str: テーブル名。
        """
        if table_name is None:
            symbol = self.config_manager.get("SYMBOL")
            interval = self.config_manager.get("INTERVAL")
            self.table_name = f'{symbol}_{interval}_market_data'
            self.table_name = self.table_name.lower()
            return self.table_name
        self.table_name = table_name
        #すべてのテーブル名は小文字である必要がある
        self.table_name = self.table_name.lower()
        return self.table_name

    def make_table_name_tech(self,table_name=None)->str:
        """
        技術指標を含むデータを格納するテーブル名を生成または設定します。

        Args:
            table_name (Optional[str]): テーブル名。指定しない場合は、`make_table_name` メソッドで生成されたテーブル名に '_tech' を追加して使用されます。

        Returns:
            str: テーブル名。
        """
        if table_name is None:
            self.table_name = self.make_table_name() + '_tech'
            self.table_name = self.table_name.lower()
            return self.table_name
        self.table_name = table_name
        #すべてのテーブル名は小文字である必要がある
        self.table_name = self.table_name.lower()
        return self.table_name

    def create_table(self):
        """
        データベース内に新しいテーブルを作成します。テーブル名はインスタンス変数 `table_name` に基づきます。
        テーブルが既に存在する場合は何もしません。テーブルの構造は市場データの記録に特化しています。

        テーブル構造:
        - start_at: データの開始時刻 (TIMESTAMP型)
        - open: 始値 (FLOAT型)
        - high: 高値 (FLOAT型)
        - low: 安値 (FLOAT型)
        - close: 終値 (FLOAT型)
        - volume: 取引量 (FLOAT型)
        - turnover: 売上高 (FLOAT型)
        - date: 日付 (TEXT型)
        - funding_rate: ファンディングレート (FLOAT型)
        - p_close: プレミアムクローズ (FLOAT型)
        - oi: オープンインタレスト (FLOAT型)

        テーブル作成後、成功メッセージをログに記録します。失敗した場合は、エラーメッセージをログに記録し、トランザクションをロールバックします。
        """
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = '{self.table_name}'
                    );
                """)
                exists = cursor.fetchone()[0]

                if not exists:
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.table_name} (
                            start_at TIMESTAMP,
                            open FLOAT,
                            high FLOAT,
                            low FLOAT,
                            close FLOAT,
                            volume FLOAT,
                            turnover FLOAT,
                            date TEXT
                        );
                    """)
                    self.conn.commit()
                    self.logger.log_system_message(f"Table '{self.table_name}' created.")
            except psycopg2.Error as e:
                self.logger.log_system_message(f"Failed to create table '{self.table_name}': {e}")
                self.conn.rollback()

    def load_data_for_db(self,csvfilename)->pd.DataFrame:
        """
        指定されたCSVファイル名からデータをロードするか、既にロードされている生データを返します。

        Args:
            csvfilename (str): データをロードするCSVファイルの名前。Noneの場合、既にロードされている生データを返します。

        Returns:
            pd.DataFrame: ロードされた生データ。
        """
        if csvfilename is None:
            raw = self.get_raw()
        else:
            raw = self.load_data_from_csv(csvfilename)
        return raw

    def convert_decimal_to_float(self, df)->pd.DataFrame:
        """
        DataFrame内のdecimal.Decimal型のデータをfloatに変換します。

        Args:
            df (pd.DataFrame): 変換するデータが含まれるDataFrame。

        Returns:
            pd.DataFrame: 変換後のDataFrame。
        """
        for column, dtype in df.dtypes.items():
            if dtype == object:
                try:
                    # decimal.Decimalを含む可能性のあるカラムをfloatに変換
                    if isinstance(df[column].iloc[0], decimal.Decimal) or df[column].isnull().any():
                        df[column] = df[column].apply(lambda x: float(x) if x is not None else None)
                except IndexError:
                    # カラムが空の場合は何もしない
                    pass
        return df

    def load_data_from_period(self, start_date, end_date, table_name=None)->pd.DataFrame:
        """
        指定された期間内のデータをデータベースからロードします。テーブル名も指定可能です。

        Args:
            start_date (str): データの開始日 (YYYY-MM-DD形式)。
            end_date (str): データの終了日 (YYYY-MM-DD形式)。
            table_name (Optional[str]): データをロードするテーブル名。指定されていない場合、インスタンス変数のテーブル名を使用。

        Returns:
            pd.DataFrame: ロードされたデータ。
        """
        # テーブル名が指定されていない場合は、インスタンスのテーブル名を使用
        target_table_name = table_name or self.table_name

        # 指定された期間に対応するデータを選択するSQLクエリを定義
        query = f"""
        SELECT * FROM {target_table_name}
        WHERE date >= %s AND date <= %s
        ORDER BY start_at ASC;
        """

        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query, (start_date, end_date))
                records = cursor.fetchall()
                raw = pd.DataFrame(records, columns=[col.name for col in cursor.description])

                # decimal.Decimal型のデータをfloatに変換
                raw = self.convert_decimal_to_float(raw)

                self.set_raw(raw)
                return raw
        except psycopg2.Error as e:
            self.logger.log_system_message(f"指定された期間 '{start_date} から {end_date}' のデータを {target_table_name} からロードするのに失敗しました: {e}")
            return None

    def load_data_from_datetime_period(self, start_datetime, end_datetime, table_name=None)->pd.DataFrame:
        """
        指定された日時期間内のデータをデータベースからロードします。テーブル名も指定可能です。

        Args:
            start_datetime (str): データの開始日時 (YYYY-MM-DD HH:MM:SS形式)。
            end_datetime (str): データの終了日時 (YYYY-MM-DD HH:MM:SS形式)。
            table_name (Optional[str]): データをロードするテーブル名。指定されていない場合、インスタンス変数のテーブル名を使用。

        Returns:
            pd.DataFrame: ロードされたデータ。
        """
        # テーブル名が指定されていない場合は、インスタンスのテーブル名を使用
        target_table_name = table_name or self.table_name

        # 指定された日時期間に対応するデータを選択するSQLクエリを定義
        query = f"""
        SELECT * FROM {target_table_name}
        WHERE start_at >= %s AND start_at <= %s
        ORDER BY start_at ASC;
        """

        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query, (start_datetime, end_datetime))
                records = cursor.fetchall()
                raw = pd.DataFrame(records, columns=[col.name for col in cursor.description])

                # decimal.Decimal型のデータをfloatに変換
                raw = self.convert_decimal_to_float(raw)

                self.set_raw(raw)
                return raw
        except psycopg2.Error as e:
            self.logger.log_system_message(f"指定された日時期間 '{start_datetime} から {end_datetime}' のデータを {target_table_name} からロードするのに失敗しました: {e}")
            return None

    def load_data_from_db(self, table_name=None)->pd.DataFrame:
        """
        データベースからデータを読み込み、'start_at'で昇順に並べ替えたデータを返します。

        Args:
            table_name (Optional[str]): データをロードするテーブル名。指定されていない場合、インスタンス変数のテーブル名を使用。

        Returns:
            pd.DataFrame: ロードされたデータ。
        """
        self.table_name = table_name or self.table_name
        try:
            query = f"SELECT * FROM {self.table_name} ORDER BY start_at ASC;"
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query)
                records = cursor.fetchall()
                raw = pd.DataFrame(records, columns=[col.name for col in cursor.description])

                # decimal.Decimal型のデータをfloatに変換
                raw = self.convert_decimal_to_float(raw)

                self.set_raw(raw)
                return self.get_raw()
        except psycopg2.Error as e:
            self.logger.log_system_message(f"テーブル '{self.table_name}' からデータを読み込むのに失敗しました: {e}")
            return None

    def load_recent_data_from_db(self, table_name=None,num_rows=1000)->pd.DataFrame:
        """
        データベースから最新のデータを指定された行数までロードします。テーブルが存在しない場合はエラーを返します。
        指定された行数未満のデータしかない場合は、存在するすべてのデータをロードします。

        Args:
            num_rows (int): ロードする最大行数。デフォルトは1000。
            table_name (Optional[str]): データをロードするテーブル名。指定されていない場合、インスタンス変数のテーブル名を使用。

        Returns:
            pd.DataFrame: ロードされたデータ。
        """
        # テーブル名が指定されていない場合は、インスタンスのテーブル名を使用
        self.table_name = table_name or self.table_name
        try:
            # テーブル名をダブルクオーテーションで囲まない
            #query = f"SELECT * FROM {self.table_name} ORDER BY start_at DESC LIMIT {num_rows};"
            query = "SELECT * FROM {} ORDER BY start_at DESC LIMIT %s;".format(self.table_name)
            #cursor.execute(query, (num_rows,))

            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query, (num_rows,))
                records = cursor.fetchall()

                # 結果が空でない場合、DataFrameを作成
                if records:
                    raw = pd.DataFrame(records, columns=[col.name for col in cursor.description])
                    # decimal.Decimal型のデータをfloatに変換
                    raw = self.convert_decimal_to_float(raw)
                    # 昇順にソート
                    raw = raw.sort_values(by='start_at').reset_index(drop=True)
                    self.set_raw(raw)
                    return raw
                else:
                    return pd.DataFrame()  # 空のDataFrameを返す
        except psycopg2.Error as e:
            self.logger.log_system_message(f"テーブル '{self.table_name}' からデータを読み込むのに失敗しました: {e}")
            return None


    def write_data_to_db(self,csvfilename=None,df=None):
        """
        指定されたCSVファイルまたはDataFrameからデータを読み込み、データベースに書き込みます。

        Args:
            csvfilename (Optional[str]): データをロードするCSVファイルの名前。Noneの場合、df引数を使用。
            df (Optional[pd.DataFrame]): データベースに書き込むデータが含まれるDataFrame。Noneの場合、csvfilename引数を使用。

        Raises:
            ValueError: 書き込むデータが存在しない場合。
        """
        if df is not None:
            self.set_raw(df)
            raw = df
        else:
            raw = self.load_data_for_db(csvfilename)

        if raw is None:
            raise ValueError("No data to write to database")
        # start_atカラムがUnixタイムスタンプを含むと仮定して、datetimeに変換
        if 'start_at' in raw.columns:
            raw['start_at'] = pd.to_datetime(raw['start_at'], unit='s')  # Unixタイムスタンプをdatetimeに変換

        try:
            with self.conn.cursor() as cursor:
                df_columns = ['"' + column.replace('"', '""') + '"' for column in raw.columns]
                columns = ', '.join(df_columns)
                values = [tuple(row) for row in raw.itertuples(index=False, name=None)]
                insert_query = f"INSERT INTO {self.table_name} ({columns}) VALUES %s"
                extras.execute_values(cursor, insert_query, values)
                self.conn.commit()
        except psycopg2.Error as e:
            self.logger.log_system_message(f"Failed to write data to table '{self.table_name}': {e}")
            self.conn.rollback()

    def insert_new_data(self,csvfilename=None,df=None):
        """
        'start_at'列の重複がない場合に新しいデータをデータベースに挿入します。

        Args:
            csvfilename (Optional[str]): データをロードするCSVファイルの名前。Noneの場合、df引数を使用。
            df (Optional[pd.DataFrame]): データベースに挿入するデータが含まれるDataFrame。Noneの場合、csvfilename引数を使用。
        """
        if df is not None:
            self.set_raw(df)
            new_data = df
        else:
            new_data = self.load_data_for_db(csvfilename)

        # 'start_at'が既にdatetimeでない場合はdatetimeに変換します
        if 'start_at' in new_data.columns and new_data['start_at'].dtype == 'float64':
            new_data['start_at'] = pd.to_datetime(new_data['start_at'], unit='s')

        # 重複をチェックして新しいデータを挿入します
        new_data_records = new_data.to_dict('records')
        with self.conn.cursor() as cursor:
            for record in new_data_records:
                try:
                    cursor.execute(f"""
                        SELECT EXISTS (
                            SELECT 1 FROM {self.table_name} WHERE start_at = %s
                        )
                    """, (record['start_at'],))
                    exists = cursor.fetchone()[0]

                    if not exists:
                        columns = ', '.join(['"' + col + '"' for col in new_data.columns])
                        placeholders = ', '.join(['%s'] * len(record))
                        cursor.execute(f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})", list(record.values()))
                        self.logger.log_verbose_message(f"New data inserted: {record['date']} {record['start_at']}")
                except psycopg2.Error as e:
                    self.logger.log_system_message(f"データ挿入に失敗しました: {e}")
                    self.conn.rollback()
        self.conn.commit()


    def is_data_exist(self,table_name=None)->bool:
        """
        指定されたテーブルがデータベース内に存在するかどうかを確認します。

        Args:
            table_name (Optional[str]): 確認するテーブル名。指定されていない場合、インスタンス変数のテーブル名を使用。

        Returns:
            bool: テーブルが存在する場合はTrue、存在しない場合はFalse。
        """
        if table_name is not None:
            self.make_table_name(table_name)

        with self.conn.cursor() as cursor:
            cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = lower(%s)
                    );
                """, (self.table_name,))
            exists = cursor.fetchone()[0]
            return exists == True or exists == 't'  # PostgreSQLの戻り値に合わせて適宜調整


    def import_to_db(self,csvfilename=None,dataframe=None,table_name=None):
        """
        CSVファイルまたはDataFrameからデータをインポートし、データベースに挿入します。テーブルが存在しない場合は、新たに作成します。

        Args:
            csvfilename (Optional[str]): データをロードするCSVファイルの名前。Noneの場合、dataframe引数を使用。
            dataframe (Optional[pd.DataFrame]): データベースに挿入するデータが含まれるDataFrame。Noneの場合、csvfilename引数を使用。
            table_name (Optional[str]): データを挿入するテーブル名。指定されていない場合、インスタンス変数のテーブル名を使用。
        """
        if self.is_data_exist(table_name):
            self.insert_new_data(csvfilename,dataframe)
        else:
            self.create_table_from_df(dataframe, table_name)
            self.write_data_to_db(csvfilename,dataframe)

    def create_table_from_df(self, df, table_name=None):
        """
        DataFrameのカラム情報からデータベースのテーブルを作成します。'start_at'カラムは'TIMESTAMP'型として扱います。

        Args:
            df (pd.DataFrame): テーブル作成の基になるデータが含まれるDataFrame。
            table_name (Optional[str]): 作成するテーブル名。指定されていない場合、インスタンス変数のテーブル名を使用。
        """
        if table_name is not None:
            self.table_name = table_name

        column_definitions = []
        for column, dtype in df.dtypes.items():
            sql_dtype = dtype_mapping(str(dtype), column)  # カラム名も渡す
            column_definitions.append(f"{column} {sql_dtype}")
        columns_sql = ', '.join(column_definitions)

        create_table_sql = f"CREATE TABLE IF NOT EXISTS {self.table_name} ({columns_sql});"

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(create_table_sql)
                self.conn.commit()
                self.logger.log_verbose_message(f"Table '{self.table_name}' created with columns: {', '.join(column_definitions)}")
            except psycopg2.Error as e:
                self.logger.log_verbose_message(f"Failed to create table '{self.table_name}': {e}")
                self.conn.rollback()

    def get_latest_data(self, table_name=None)->pd.DataFrame:
        """
        指定されたテーブルから最新のデータ行を取得します。

        Args:
            table_name (Optional[str]): データを取得するテーブル名。指定されていない場合、インスタンス変数のテーブル名を使用。

        Returns:
            pd.DataFrame: 最新のデータ行を含むDataFrame。
        """
        new_df = self.get_latest_data_sub(table_name)
        all_df = self.get_raw()
        latest_df = all_df.iloc[-1:]
        if new_df[COLUMN_START_AT].iloc[0] == latest_df[COLUMN_START_AT].iloc[-1]:
            self.logger.log_verbose_message("最新のデータはすでにデータベースにあります。")
            self.logger.log_verbose_message(latest_df)
            return None

        cols_to_add = set(latest_df.columns) - set(new_df.columns)

        # 識別されたカラムに対してデフォルト値を設定してDataFrame Bを拡張
        for col in cols_to_add:
            # カラムが文字列型の場合
            if latest_df[col].dtype == object:
                new_df[col] = None
            # カラムが浮動小数点数型の場合
            elif np.issubdtype(latest_df[col].dtype, np.floating):
                new_df[col] = 0.0
            else:
            # その他の型に対しては適切な処理を実施（例では扱わない）
                pass

        # DataFrame AとBを結合
        df_a_updated = pd.concat([all_df, new_df], ignore_index=True)
        self.set_raw(df_a_updated)
        self.logger.log_verbose_message(f"最新のデータを追加しました。")
        self.logger.log_verbose_message(df_a_updated)
        return df_a_updated


    def get_latest_data_sub(self, table_name):
        """指定されたテーブルから最新のデータ行を取得します。

        Args:
            table_name (str): データを取得するテーブル名。

        Returns:
            pd.DataFrame: 最新のデータ行を含むDataFrame。
        """
        # テーブル名が指定されていない場合は、インスタンスのテーブル名を使用
        self.table_name = table_name or self.table_name
        query = f"""SELECT * FROM {table_name} ORDER BY start_at DESC LIMIT 1;"""

        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query)
                record = cursor.fetchone()
                if record:
                    raw = pd.DataFrame([record], columns=[col.name for col in cursor.description])
                    raw = self.convert_decimal_to_float(raw)
                    return raw
                else:
                    return pd.DataFrame()  # 空のDataFrameを返す
        except psycopg2.Error as e:
            self.logger.log_system_message(f"テーブル '{table_name}' から最新のデータを取得するのに失敗しました: {e}")
            return None







