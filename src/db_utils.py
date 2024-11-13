import pandas as pd
import sqlite3
import config


def get_all_states(data_path=config.SQL_LITE_FILE_NAME):
    conn = sqlite3.connect(data_path)
    query = """SELECT DISTINCT ca_state from customer_address"""
    all_states = pd.read_sql(query, conn)
    conn.close()
    return all_states['ca_state'].values


def get_all_item_class(data_path=config.SQL_LITE_FILE_NAME):
    conn = sqlite3.connect(data_path)
    query = """SELECT DISTINCT i_class from item"""
    all_states = pd.read_sql(query, conn)
    conn.close()
    return all_states['i_class'].values


def get_all_item_categories(data_path=config.SQL_LITE_FILE_NAME):
    conn = sqlite3.connect(data_path)
    query = """SELECT DISTINCT i_category from item"""
    all_states = pd.read_sql(query, conn)
    conn.close()
    return all_states['i_category'].values
