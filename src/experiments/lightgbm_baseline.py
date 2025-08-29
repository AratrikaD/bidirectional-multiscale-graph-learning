from lightgbm import LGBMRegressor
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler

import matplotlib.pyplot as plt


def run_lgbm_experiment(data_path, output_path):
# Load Data
    neighborhood_df = pd.read_csv(f"{data_path}/all_neighborhood_features_rotterdam.csv")
    transaction_df = pd.read_csv(f"{data_path}/synthetic_transactions.csv")


    # Preprocess transaction data
    transaction_df['DATUM'] = pd.to_datetime(transaction_df['DATUM'])
    transaction_df.sort_values('DATUM', inplace=True)
    transaction_df['YEAR'] = transaction_df['DATUM'].dt.year
    transaction_df['MONTH'] = transaction_df['DATUM'].dt.month
    transaction_df.drop(["DATUM"], axis=1, inplace=True)

    # Combine data
    def combine_data(transactions, node_features):
        combined_df = transactions.merge(node_features, how="left", on=["BUURTCODE", "YEAR"])
        return combined_df

    combined_df = combine_data(transactions=transaction_df, node_features=neighborhood_df)

    # Add DATE column and sort
    combined_df["DATE"] = pd.to_datetime(combined_df["YEAR"].astype(str) + "-" + combined_df["MONTH"].astype(str))
    combined_df = combined_df.sort_values("DATE")

    # Feature selection
    features = [col for col in combined_df.columns if col not in ["LOG_KOOPSOM", "DATE", "TRANSID"]]
    target = "LOG_KOOPSOM"

    # LightGBM with Sliding Windows
    min_date = combined_df["DATE"].min()
    max_date = combined_df["DATE"].max()
    start_train = min_date
    end_train = start_train + pd.DateOffset(months=60)
    test_month = end_train + pd.DateOffset(months=1)

    model = None
    scaler = StandardScaler()
    all_window_preds = []
    epoch_stats = []
    prev_booster = None

    while test_month <= max_date:
        print("Train Start Date: ", start_train)
        print("Train End Date: ", end_train)
        print("Test Month", test_month)

        train_data = combined_df[(combined_df["DATE"] >= start_train) & (combined_df["DATE"] <= end_train)]
        test_data = combined_df[combined_df["DATE"] == test_month]
        print(len(test_data), "test data points")
        if test_data.empty:
            print("empty")
            break

        base_params = {'objective': 'L2', 'n_estimators': 500, 'learning_rate': 0.05,
                    'num_leaves': 1000, 'min_data_in_leaves': 10, 'feature_fraction': 0.7, 'max_depth': 75,
                    'lambda_l2': 10e-5, 'path_smooth': 10e-5, 'n_jobs': -1}

        if model is None:
            print("Training new model...")
            model = LGBMRegressor(**base_params, verbose=-1)
            model.fit(train_data[features], train_data[target])
            prev_booster = model.booster_
        else:
            print("Updating model with new data...", flush=True)
            new_params = {**base_params, "n_estimators": 50}
            model = LGBMRegressor(**new_params, verbose=-1)
            model.fit(
                train_data[features],
                train_data[target],
                init_model=prev_booster
            )
            prev_booster = model.booster_

        # Make predictions
        predictions = model.predict(test_data[features])
        actuals = test_data[target].values

        # Train predictions for metrics
        train_preds = model.predict(train_data[features])
        train_mse = np.mean((train_data[target] - train_preds) ** 2)
        train_mape = np.mean(np.abs((np.exp(train_data[target]) - np.exp(train_preds)) / np.exp(train_data[target]))) * 100

        print(f"Train MSE: {train_mse}")
        print(f"Train MAPE: {train_mape}")

        mse = np.mean((actuals - predictions) ** 2)
        print(f"MSE: {mse}")
        mape = np.mean(np.abs((np.exp(actuals) - np.exp(predictions)) / np.exp(predictions))) * 100
        print(f"MAPE: {mape}")

        epoch_stats.append({
            "window_start": start_train.strftime('%Y-%m'),
            "epoch": 1,
            "train_mape": train_mape,
            "test_mape": mape,
            "train_mse": train_mse,
            "test_mse": mse,
        })
        preds_df = pd.DataFrame({
            "window_start": start_train.strftime('%Y-%m'),
            "BUURTCODE": test_data["BUURTCODE"].values,
            "YEAR": test_data["YEAR"].values,
            "MONTH": test_data["MONTH"].values,
            "TRANSID": test_data["TRANSID"].values,
            "y_true": actuals,
            "y_pred": predictions,
        })

        all_window_preds.append(preds_df)
        # Move window forward
        print("current start:", start_train)
        start_train += pd.DateOffset(months=1)
        print("updated start", start_train)
        end_train += pd.DateOffset(months=1)
        test_month += pd.DateOffset(months=1)
        print("--------------------------------------------------")
        print("Updated Train Start Date: ", start_train)
        print("Updated Train End Date: ", end_train)
        print("Updated Test Month: ", test_month)
        print("--------------------------------------------------")

    # Save predictions and stats
    final_preds_df = pd.concat(all_window_preds, ignore_index=True)
    final_preds_df.to_csv(f"./outputs/all_test_predictions_ml_synth.csv", index=False)

    stats_df = pd.DataFrame(epoch_stats)
    stats_df.to_csv(f"./outputs/training_stats_ml_synth.csv", index=False)

    # Online Learning Approach (example, not run above)
    # df = combined_df
    # scaler = RobustScaler()
    # params = {'objective': 'L2', 'n_estimators': 1000, 'learning_rate': 0.05,
    #           'num_leaves': 1000, 'min_data_in_leaves': 10, 'feature_fraction': 0.7, 'max_depth': 75,
    #           'lambda_l2': 10e-5, 'path_smooth': 10e-5, 'n_jobs': -1}
    # model = LGBMRegressor(**params, verbose=-1)
    # min_date = df["DATE"].min()
    # max_date = df["DATE"].max()
    # start_train = min_date
    # test_month = start_train + pd.DateOffset(months=60)
    # predictions = []
    # actuals = []
    # while test_month <= max_date:
    #     train_idx = df["DATE"] < test_month
    #     test_idx = df["DATE"] == test_month
    #     print("Test Month:", test_month)
    #     X_train, y_train = df.loc[train_idx, features], df.loc[train_idx, target]
    #     X_test, y_test = df.loc[test_idx, features], df.loc[test_idx, target]
    #     if X_train.empty or X_test.empty:
    #         break
    #     if start_train == min_date:
    #         model.fit(X_train, y_train)
    #     else:
    #         model.set_params(n_estimators=model.n_estimators + 20)
    #         model.fit(X_train, y_train, init_model=model)
    #     preds = model.predict(X_test)
    #     predictions.extend(preds)
    #     actuals.extend(y_test.values)
    #     rmse = root_mean_squared_error(actuals, predictions)
    #     print(f"RMSE: {rmse}")
    #     mape = np.mean(np.abs((np.exp(actuals) - np.exp(predictions)) / np.exp(predictions))) * 100
    #     print(f"MAPE: {mape}")
    #     test_month += pd.DateOffset(months=1)
    # rmse = root_mean_squared_error(actuals, predictions)
    # print(f"Final RMSE: {rmse}")
    # mape = np.mean(np.abs((np.exp(actuals) - np.exp(predictions)) / np.exp(predictions))) * 100
    # print(f"Final MAPE: {mape}")