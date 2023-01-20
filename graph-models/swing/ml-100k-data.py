import pandas as pd

train_data = pd.read_csv('./ua.base', sep="\t", engine="python", names=["userid", "movieid", "rate", "event_timestamp"])
# test_data = pd.read_csv(test_path, sep="\t", engine="python", names=["userid", "movieid", "rate", "event_timestamp"])

print(train_data.loc[:, ['userid','movieid']])
train_data.loc[:, ['userid','movieid']].to_csv('./ml-100k-uabase.csv')
