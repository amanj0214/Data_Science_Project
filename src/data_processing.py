import pandas as pd


def process_data(data):
    data['return_rate'] = data['wr_return_quantity'] / data['ws_quantity']

    lag_values = data.sort_values(['d_year', 'd_moy']).groupby(
        ['ca_state', 'i_class', 'i_category']
    ).shift(1)
    lag_values.columns = ['lag_1_' + c for c in lag_values.columns]
    processed_data = pd.concat([data, lag_values], axis=1).sort_values(
        ['ca_state', 'i_class', 'i_category', 'd_year', 'd_moy']).reset_index(drop=True)
    processed_data = processed_data.dropna()
    return processed_data


if __name__ == "__main__":
    print('')
