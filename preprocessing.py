import opendatasets as od
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class SoftIto_prework:

    def download_the_dataset():
        od.download('https://www.kaggle.com/jsphyg/weather-dataset-rattle-package')
        raw_df = pd.read_csv('weather-dataset-rattle-package/weatherAUS.csv')
        raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)
        return raw_df

    def create_training_validation_test_sets(raw_df):
        year = pd.to_datetime(raw_df.Date).dt.year
        train_df, val_df, test_df = raw_df[year < 2015], raw_df[year == 2015], raw_df[year > 2015]
        return train_df,val_df,test_df

    def create_inputs_and_targets(train_df, val_df, test_df):
        input_cols = list(train_df.columns)[1:-1]
        target_col = 'RainTomorrow'
        train_inputs, train_targets = train_df[input_cols].copy(), train_df[target_col].copy()
        val_inputs, val_targets = val_df[input_cols].copy(), val_df[target_col].copy()
        test_inputs, test_targets = test_df[input_cols].copy(), test_df[target_col].copy()
        return target_col,train_inputs,train_targets,val_inputs,val_targets,test_inputs,test_targets

    def identify_numeric_and_categorical_columns(train_inputs):
        numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()[:-1]
        categorical_cols = train_inputs.select_dtypes('object').columns.tolist()
        return numeric_cols,categorical_cols

    def scale_numeric_features(raw_df, train_inputs, val_inputs, test_inputs, numeric_cols):
        scaler = MinMaxScaler().fit(raw_df[numeric_cols])
        train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
        val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
        test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

    def impute_missing_numerical_values(raw_df, train_inputs, val_inputs, test_inputs, numeric_cols):
        imputer = SimpleImputer(strategy = 'mean').fit(raw_df[numeric_cols])
        train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
        val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
        test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

    def one_hot_encode_categorical_features(raw_df, train_inputs, val_inputs, test_inputs, categorical_cols):
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(raw_df[categorical_cols])
        encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
        train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
        val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
        test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

    def save_processed_data_to_disk(train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets):
        train_inputs.to_parquet('train_inputs.parquet')
        val_inputs.to_parquet('val_inputs.parquet')
        test_inputs.to_parquet('test_inputs.parquet')
        pd.DataFrame(train_targets).to_parquet('train_targets.parquet')
        pd.DataFrame(val_targets).to_parquet('val_targets.parquet')
        pd.DataFrame(test_targets).to_parquet('test_targets.parquet')

    def load_processed_data_from_disk(target_col):
        train_inputs = pd.read_parquet('train_inputs.parquet')
        val_inputs = pd.read_parquet('val_inputs.parquet')
        test_inputs = pd.read_parquet('test_inputs.parquet')
        train_targets = pd.read_parquet('train_targets.parquet')[target_col]
        val_targets = pd.read_parquet('val_targets.parquet')[target_col]
        test_targets = pd.read_parquet('test_targets.parquet')[target_col]

softIto_prework = SoftIto_prework() 

raw_df = softIto_prework.download_the_dataset()
train_df, val_df, test_df = softIto_prework.create_training_validation_test_sets(raw_df)
target_col, train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets = softIto_prework.create_inputs_and_targets(train_df, val_df, test_df)
# = create_inputs_and_targets(train_df, val_df, test_df)
numeric_cols, categorical_cols = softIto_prework.identify_numeric_and_categorical_columns(train_inputs)
softIto_prework.scale_numeric_features(raw_df, train_inputs, val_inputs, test_inputs, numeric_cols)
softIto_prework.impute_missing_numerical_values(raw_df, train_inputs, val_inputs, test_inputs, numeric_cols)
softIto_prework.one_hot_encode_categorical_features(raw_df, train_inputs, val_inputs, test_inputs, categorical_cols)
softIto_prework.save_processed_data_to_disk(train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets)
softIto_prework.load_processed_data_from_disk(target_col)