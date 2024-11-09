import pandas as pd
import sqlite3
import src.config as config


def get_all_states(data_path=config.SQL_LITE_FILE_NAME):
    conn = sqlite3.connect(data_path)
    query = """SELECT DISTINCT ca_state from customer_address"""
    all_states = pd.read_sql(query, conn)
    conn.close()
    return all_states['ca_state'].values
