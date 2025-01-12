import pandas as pd


def channels_by_row_index(path_data, filename):
    df = pd.read_csv(path_data)
    df_melted = pd.melt(df, var_name='metric', value_name='value')
    df_melted[['channel', 'measurement']] = df_melted['metric'].str.extract(r'channel_(\d+)_(.+)')
    df_melted.drop(columns=['metric'], inplace=True)
    df_pivoted = df_melted.pivot_table(index=['channel'], columns='measurement', values='value',
                                       aggfunc='first').reset_index()
    label_columns = [col for col in df.columns if 'label' in col]
    label_df = df[label_columns]
    final_df = pd.concat([df_pivoted, label_df.reset_index(drop=True)], axis=1)
    final_df.drop(columns=['channel'], inplace=True)
    final_df.to_csv(filename, index=False)

