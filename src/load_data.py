import pandas as pd
import sqlite3
import config


def load_data(data_path=config.SQL_LITE_FILE_NAME):
    conn = sqlite3.connect(data_path)
    query = """
    SELECT 
        d_year + 20 AS d_year,
        d_moy,
        ca_state,
        i_class,
        i_category,
        SUM(ws_quantity) AS ws_quantity,
        SUM(wr_return_quantity) AS wr_return_quantity,
        SUM(wr_net_loss) AS wr_net_loss
    FROM web_sales ws 
    JOIN item i ON ws.ws_item_sk = i.i_item_sk 
    JOIN date_dim dd ON dd.d_date_sk = ws.ws_sold_date_sk 
    JOIN web_returns wr ON wr.wr_order_number = ws.ws_order_number 
    JOIN customer_address ca ON wr.wr_returning_addr_sk = ca.ca_address_sk 
    WHERE d_year IS NOT NULL AND d_moy IS NOT NULL 
      AND i_class != 'None' AND i_category != 'None'
      -- TODO remove below filters
      AND ca_state = 'CA'
    GROUP BY d_year, d_moy, ca_state, i_class, i_category
    HAVING SUM(wr_net_loss) > AVG(wr_net_loss)
    """
    data_raw = pd.read_sql(query, conn)
    conn.close()
    return data_raw


if __name__ == "__main__":
    data = load_data()
    print(data.head())