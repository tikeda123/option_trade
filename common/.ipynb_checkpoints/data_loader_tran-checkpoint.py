import numpy as np
import psycopg2
from psycopg2 import extras
from psycopg2.extensions import register_adapter, AsIs
register_adapter(np.int64, psycopg2._psycopg.AsIs)
import pandas as pd
import os, sys

from datetime import datetime, timedelta, timezone

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.data_loader_db import DataLoaderDB


class DataLoaderTransactionDB(DataLoaderDB):
    def __init__(self, table_name=None):
        super().__init__(table_name)

    def create_table(self, table_name, table_type, is_aggregated=False):
        #self._drop_table_if_exists(table_name)
        if table_type == 'fxaccount':
            self._create_table_fxaccount(table_name,is_aggregated)
        elif table_type == 'fxtransaction':
            self._create_table_fxtransaction(table_name,is_aggregated)
        elif table_type == 'trade_log':
            self.create_table_trade_log(table_name,is_aggregated)

    def _convert_timestamp_nat_to_none(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame's NaT values to '1960-01-01 00:00:00' for TIMESTAMP columns and NaN values to None for other types.
        This function will replace all NaT values with '1960-01-01 00:00:00' in TIMESTAMP columns and NaN or NaT values with None in other columns, making it suitable for database insertion where these missing values need to be explicitly represented as NULL or a default TIMESTAMP.
        """
        for column in df.columns:
            # TIMESTAMP型の列のNaTを'1960-01-01 00:00:00'に置き換える
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                df[column] = df[column].fillna(pd.Timestamp('1960-01-01 00:00:00'))
            else:
                # その他の型のNaNをNoneに置き換える
                df[column] = df[column].apply(lambda x: None if pd.isna(x) else x)

        return df


    def write_db_aggregated_table(self, df: pd.DataFrame, table_name):
        table_name = f"{table_name}_aggregated"
        df = self._convert_timestamp_nat_to_none(df)
        self.write_db(df, table_name)

    def write_db(self, df: pd.DataFrame, table_name):
        df = self._convert_nat_to_none(df)
        try:
            with self.conn.cursor() as cursor:
                df_columns = ['"' + column.replace('"', '""') + '"' for column in df.columns]
                columns = ', '.join(df_columns)
                values = [tuple(row) for row in df.itertuples(index=False, name=None)]
                insert_query = f"INSERT INTO {table_name} ({columns}) VALUES %s"
                extras.execute_values(cursor, insert_query, values)
                self.conn.commit()
        except psycopg2.Error as e:
            self.logger.log_system_message(f"Failed to write data to table '{table_name}': {e}")
            self.conn.rollback()

    def drop_table_if_exists(self, table_name):
        with self.conn.cursor() as cursor:
            try:
                # テーブルが存在するかどうかを確認
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = lower(%s)
                    );
                """, (table_name,))
                exists_table = cursor.fetchone()[0]

                # テーブルが存在する場合、削除（CASCADEオプションを削除）
                if exists_table:
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
                    self.logger.log_system_message(f"Table '{table_name}' dropped.")
                    # ここでは、テーブルのみが削除され、関連するシーケンスは削除されない

                self.conn.commit()
            except psycopg2.Error as e:
                self.logger.log_system_message(f"Failed to drop table '{table_name}': {e}")
                self.conn.rollback()


    def get_next_serial(self, table_name):
        sequence_name = f"{table_name}_serial_seq"
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(f"SELECT nextval('{sequence_name}');")
                next_serial = cursor.fetchone()[0]
                return next_serial
            except psycopg2.Error as e:
                self.logger.log_system_message(f"Failed to retrieve next serial from sequence '{sequence_name}': {e}")
                self.conn.rollback()
                return None

    def _create_sequence_if_not_exists(self, sequence_name):
        with self.conn.cursor() as cursor:
            try:
                # SEQUENCEの存在確認
                cursor.execute(f"SELECT to_regclass('{sequence_name}') IS NOT NULL;")
                sequence_exists = cursor.fetchone()[0]

                # SEQUENCEが存在しない場合、作成
                if not sequence_exists:
                    cursor.execute(f"CREATE SEQUENCE {sequence_name};")
                    self.conn.commit()
                    self.logger.log_system_message(f"Sequence '{sequence_name}' created.")
                else:
                    self.logger.log_system_message(f"Sequence '{sequence_name}' already exists. No action taken.")
            except psycopg2.Error as e:
                self.logger.log_system_message(f"Failed to create sequence '{sequence_name}': {e}")
                self.conn.rollback()


    def _create_table_fxaccount(self, table_name, is_aggregated=False):
        sequence_name = f"{table_name}_serial_seq"

        if is_aggregated:
            table_name = f"{table_name}_aggregated"

        if self.is_data_exist(table_name):
            print(f"Table '{table_name}' already exists.")
            return

        self._create_sequence_if_not_exists(sequence_name)

        with self.conn.cursor() as cursor:
            try:
                # 既存のSEQUENCEを利用してテーブルを作成
                cursor.execute(f"""
                    CREATE TABLE {table_name} (
                        serial INTEGER PRIMARY KEY DEFAULT nextval('{sequence_name}'),
                        date TIMESTAMP,
                        cash_in FLOAT,
                        cash_out FLOAT,
                        amount FLOAT,
                        startup_flag INTEGER
                    );
                """)
                self.conn.commit()
                # ここではSEQUENCEの作成について言及しない
                self.logger.log_system_message(f"Table '{table_name}' created with existing sequence '{sequence_name}'.")
            except psycopg2.Error as e:
                # SEQUENCEが存在しない場合のエラーを考慮したエラーメッセージ
                self.logger.log_system_message(f"Failed to create table '{table_name}': {e}. Ensure that sequence '{sequence_name}' exists.")
                self.conn.rollback()


    def _create_table_fxtransaction(self, table_name,is_aggregated=False):
        sequence_name = f"{table_name}_serial_seq"

        if is_aggregated:
            table_name = f"{table_name}_aggregated"

        if self.is_data_exist(table_name):
            print(f"Table '{table_name}' already exists.")
            return

        self._create_sequence_if_not_exists(sequence_name)

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(f"""
                    CREATE TABLE {table_name} (
                        serial INTEGER PRIMARY KEY DEFAULT nextval('{sequence_name}'),
                        init_equity FLOAT,
                        equity FLOAT,
                        leverage FLOAT,
                        contract TEXT,
                        qty FLOAT,
                        entry_price FLOAT,
                        losscut_price FLOAT,
                        exit_price FLOAT,
                        limit_price FLOAT,
                        pl FLOAT,
                        pred INTEGER,
                        tradetype TEXT,
                        stage TEXT,
                        losscut TEXT,
                        entrytime TIMESTAMP,
                        exittime TIMESTAMP,
                        direction TEXT,
                        startup_flag INTEGER
                    );
                """)
                self.conn.commit()
                self.logger.log_system_message(f"Table '{table_name}' created.")
            except psycopg2.Error as e:
                self.logger.log_system_message(f"Failed to create table '{table_name}': {e}")
                self.conn.rollback()

    def create_table_trade_log(self, table_name, is_aggregated=False):

        sequence_name = f"{table_name}_serial_seq"

        if is_aggregated:
            table_name = f"{table_name}_aggregated"

        if self.is_data_exist(table_name):
            print(f"Table '{table_name}' already exists.")
            return

        self._create_sequence_if_not_exists(sequence_name)

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(f"""
                    CREATE TABLE {table_name} (
                        serial INTEGER PRIMARY KEY DEFAULT nextval('{sequence_name}'),
                        date TIMESTAMP,
                        message TEXT
                    );
                """)
                self.conn.commit()
                self.logger.log_system_message(f"Table '{table_name}' and  created.")
            except psycopg2.Error as e:
                self.logger.log_system_message(f"Failed to create table '{table_name}': {e}")
                self.conn.rollback()

    def update_db_by_serial(self, table_name: str, df: pd.DataFrame, serial: int):
        if not df.empty:
            df = self._convert_nat_to_none(df)
            try:
                with self.conn.cursor() as cursor:
                    set_clause = ', '.join([f"{column} = %s" for column in df.columns if column != 'serial'])
                    values = [df.iloc[0][column] for column in df.columns if column != 'serial']
                    update_query = f"""
                    UPDATE {table_name}
                    SET {set_clause}
                    WHERE serial = %s;
                    """
                    values.append(serial)
                    cursor.execute(update_query, values)
                    self.conn.commit()
                    self.logger.log_system_message(f"Data updated in table '{table_name}' for serial {serial}.")
            except psycopg2.Error as e:
                self.logger.log_system_message(f"Failed to update data in table '{table_name}': {e}")
                self.conn.rollback()
                exit()
        else:
            self.logger.log_system_message("DataFrame is empty. No data to update.")

    def _convert_nat_to_none(self, df):
        return df.apply(lambda col: col.map(lambda x: None if pd.isna(x) else int(x) if isinstance(x, np.integer) else x))

    def read_db(self, table_name=None, num_rows=1000) -> pd.DataFrame:
        """
        データベースから最新のデータを指定された行数までロードしますが、結果のDataFrameはserial番号が若い順に並び替えられます。
        テーブルが存在しない場合はエラーを返します。指定された行数未満のデータしかない場合は、存在するすべてのデータをロードします。

        Args:
            table_name (str): データをロードするテーブル名。指定されていない場合、インスタンス変数のテーブル名を使用。
            num_rows (int): ロードする最大行数。デフォルトは1000。-1の場合はデータすべて読み取る

        Returns:
            pd.DataFrame: データベースから読み込んだデータのDataFrame。DataFrameはserialで昇順にソートされます。
        """
        if table_name is None:
            table_name = self.table_name
        if table_name is None:
            raise ValueError("Table name must be specified if not set during class instantiation.")

        query = "SELECT * FROM {} ORDER BY serial DESC LIMIT %s;".format(table_name)  # 最新のデータから取得
        try:
            with self.conn.cursor() as cursor:
                if num_rows == -1:
                    cursor.execute("SELECT * FROM {} ORDER BY serial DESC;".format(table_name))  # 全データを最新から取得
                else:
                    cursor.execute(query, (num_rows,))
                rows = cursor.fetchall()
                if rows:
                    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
                    df = df.sort_values(by='serial').reset_index(drop=True)  # serialで昇順にソート
                    return df
                else:
                    return pd.DataFrame()
        except psycopg2.Error as e:
            self.logger.log_system_message(f"Failed to read data from table '{table_name}': {e}")
            self.conn.rollback()
            raise



# テスト用のコード
def main():
        # 設定ファイルのパスを指定


    # データベースへの接続
    db = DataLoaderTransactionDB()
    df = db.read_db('trading_log',num_rows=50)
    #df = db.read_db('btcusdt_fxtransaction',num_rows=3)

    print(df)

if __name__ == "__main__":
    main()
