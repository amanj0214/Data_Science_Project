import pandas as pd

def process_data(data):
    data['return_rate'] = data['wr_return_quantity'] / data['ws_quantity']
    # Lag calculation
    data = data.sort_values(['d_year', 'd_moy']).groupby(
        ['ca_state', 'i_class', 'i_category']
    ).shift(1).add_prefix('lag_1_')
    data.dropna(inplace=True)
    return data

if __name__ == "__main__":
    data = pd.read_csv("../data/processed_data.csv")
    processed_data = process_data(data)
    processed_data.to_csv("../data/processed_data.csv", index=False)
