import argparse
import pickle
import pandas as pd


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)



def read_data(filename,year, month):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    return df

def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    dicts = df[categorical].to_dict(orient='records')
    
    return dicts
    
def model_predict(dicts):
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    return y_pred
# dicts = df[categorical].to_dict(orient='records')
# X_val = dv.transform(dicts)
# y_pred = model.predict(X_val)


#df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        help="year of NYC taxi data used to predict"
    )
    parser.add_argument(
        "--month",
        help="month of NYC taxi data used to predict."
    )
    args = parser.parse_args()

    year = int(args.year)
    month = int(args.month)

    data_path = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year}-{args.month}.parquet"

    df = read_data(data_path,year, month)
    dicts = prepare_dictionaries(df)
    y_pred = model_predict(dicts)
    y_pred_avg = y_pred.mean()
    print(y_pred_avg)
