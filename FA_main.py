from Load_BatchData_txt import *
import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import os
from NewModels import modelSelection
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn import tree
from xgboost import plot_tree
from Load_BatchData_txt import Read_batch_files_fromtxt
from pathlib import Path

def moving_window(ts_data, window_size):
    ts_data_win = []
    for i in range(len(ts_data)-window_size+1):
        ts_data_win.append(ts_data[i:i+window_size])
    return ts_data_win

def data_rolling(ts_data, window_size):
    ts_data_arr = np.asarray(ts_data)
    ts_data_rol = moving_window(ts_data_arr, window_size)
    #ts_data_all_rol = np.stack((ts_hydration_data_rol, ts_temp_data_rol, ts_humid_data_rol), axis=1)
    return ts_data_rol

def date_diff(ts_date_win):
    date_diff_list = []
    for date_row in ts_date_win:
        date_a = datetime.date(int(str(date_row[-1])[:4]), int(str(date_row[-1])[5:7]), int(str(date_row[-1])[8:10]))
        date_diff_sublist = []
        for date in date_row:
            date_diff_sublist.append(0.1*(date_a - datetime.date(int(str(date)[:4]), int(str(date)[5:7]), int(str(date)[8:10]))).days)
        date_diff_list.append(date_diff_sublist[:-1])
    date_diff_list = np.stack((date_diff_list), axis=0)
    return date_diff_list

def data_features_load(file):
    df = pd.read_excel(file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # remvoe unnamed column names
    df = df.dropna(axis=1, how='all')
    all_col_name_list = df.columns.tolist()
    return all_col_name_list

def data_load1(file_list):
    df = pd.read_excel(file_list)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # remvoe unnamed column names
    df = df.dropna(axis=0, thresh=8)

    window_size = 1
    if len(df) >= 0:
        ts_date = df['Date']
        ts_temperature = df['Temperature']
        ts_humidity = df['Humidity']
        ts_hydration = df['Avg3_Hydration']
        ts_skinhealth = df['Avg3_Skinhealth']
        ts_skincareRatio = df['Skincare_Ratio']

    ts_date_win = np.stack((data_rolling(ts_date, window_size + 1)), axis=0)
    date_diff_list = date_diff(ts_date_win)
    ts_temperature_win = np.stack((data_rolling(ts_temperature, window_size)), axis=0)
    ts_humidity_win = np.stack((data_rolling(ts_humidity, window_size)), axis=0)
    ts_hydration_win = np.stack((data_rolling(ts_hydration, window_size)), axis=0)
    ts_skinhealth_win = np.stack((data_rolling(ts_skinhealth, window_size)), axis=0)
    ts_skincareRatio_win = np.stack((data_rolling(ts_skincareRatio, window_size)), axis=0)

    #raw_data = np.hstack((date_diff_list, ts_temperature_arr[:-1], ts_humidity_arr[:-1], ts_temperature_arr[1:], ts_humidity_arr[1:], ts_hydration_arr[:-1],
    #                      ts_skinhealth_arr[:-1], ts_skincareRatio_arr[1:]))

    return date_diff_list, ts_temperature_win[:-1], ts_humidity_win[:-1], ts_temperature_win[1:], ts_humidity_win[1:], \
           ts_hydration_win, ts_skinhealth_win, ts_skincareRatio_win[:-1], df

def input_output_gen(ts_date_diff_win, ts_temp_win_current, ts_humid_win_current, ts_temp_win_next, ts_humid_win_next,
                     ts_hydration_avg_win, ts_skinhealth_avg_win, ts_skincare_ratio_win):

    X_skin = np.hstack((ts_temp_win_current, ts_humid_win_current, ts_temp_win_next, ts_humid_win_next, ts_hydration_avg_win[:-1], ts_skinhealth_avg_win[:-1]))
    X_skin = np.hstack((ts_date_diff_win, X_skin, ts_skincare_ratio_win*10)) # only use the latest term of skincare ratio

    Y_hydration = np.stack((ts_hydration_avg_win[1:, -1])).reshape(-1, 1)
    Y_skinhealth = np.stack((ts_skinhealth_avg_win[1:, -1])).reshape(-1, 1)
    return X_skin, Y_hydration, Y_skinhealth

def inout_normalization(input, output):
    input = np.stack((input))
    input_nor = input / 100
    bias = np.stack((0.1*np.ones((len(input_nor), 1))), axis=0)
    input_nor_bias = np.hstack((input_nor, bias))
    output_nor = output/100
    return input_nor_bias, output_nor

def model_fitting(model, input, output):
    # XGBoost Regression Model
    #XGB_skin_model_global = xgb.XGBRegressor(n_estimators=100,learning_rate=0.1)
    model = model.fit(input, output)
    return model

def cross_validation_score_calculation(model, input, output):
    scores = cross_val_score(model, input, output, cv=50)
    print("scores = ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def data_preparation(file):
    ts_date_win, ts_temp_win_current, ts_humid_win_current, ts_temp_win_next, ts_humid_win_next, ts_hydration_avg_win, \
    ts_skinhealth_avg_win, ts_skincare_ratio_win, df = data_load1(file)
    input_data, output_data_hydration, output_data_oxygen = input_output_gen(ts_date_win, ts_temp_win_current,
                                                                             ts_humid_win_current, ts_temp_win_next,
                                                                             ts_humid_win_next, ts_hydration_avg_win,
                                                                             ts_skinhealth_avg_win, ts_skincare_ratio_win)

    input_nor_bias, output_hydration_nor = inout_normalization(input_data, output_data_hydration)
    input_nor_bias, output_oxygen_nor = inout_normalization(input_data, output_data_oxygen)
    #input_nor_bias_all.append(input_nor_bias.tolist())
    #output_hydration_nor_all.append(output_hydration_nor.tolist())
    #output_oxygen_nor_all.append(output_oxygen_nor.tolist())
    return input_nor_bias.tolist(), output_hydration_nor.tolist(), output_oxygen_nor.tolist(), df

def Error_estimation(target_data, predicttion_data):
    print("predicttion_data = ", predicttion_data)
    print("target_data.flatten() = ", target_data.flatten())
    print("((predicttion_data - target_data.flatten())/target_data.flatten()) = ", ((predicttion_data - target_data.flatten())/target_data.flatten()))
    error = (((predicttion_data - target_data.flatten())/target_data.flatten())*100)
    error_square = list(map(lambda num:num*num, error))
    error_square_std = np.std(error_square)
    error_square_mean = np.mean(error_square)
    error_std_rms = np.sqrt(error_square_std)
    error_mean_rms = np.sqrt(error_square_mean)
    return error_std_rms, error_mean_rms

def cross_validation_by_file(file_list, model_name):
    input_nor_bias_all_test = []
    output_hydration_nor_all_test = []
    output_oxygen_nor_all_test = []
    error_mean_rms_oxygen_all = []
    error_std_rms_oxygen_all = []
    error_mean_rms_hydration_all = []
    error_std_rms_hydration_all = []

    cwd = Path(os.getcwd())  # get current parent directory path
    header_path = str(cwd.parent.parent.parent.parent)
    #print("header_path = ", header_path)
    file_path_test = header_path+'/Use_Skincare/'
    txtfile_name_test = '/All_User_Files_ForCrossValidation.txt'
    test_file_list = Read_batch_files_fromtxt(header_path+txtfile_name_test)

    for file in file_list:
        input_nor_bias_all_train = []
        output_hydration_nor_all_train = []
        output_oxygen_nor_all_train = []

        print("test_file = ", file)
        input_nor_bias_test, output_hydration_nor_test, output_oxygen_nor_test, df = data_preparation(file)

        input_nor_bias_test = np.vstack((input_nor_bias_test))
        output_hydration_nor_test = np.vstack((output_hydration_nor_test))
        output_oxygen_nor_test = np.vstack((output_oxygen_nor_test))

        file_list_copy = file_list.copy()
        file_list_copy.remove(file)
        for train_file in test_file_list:
            print("train_file = ", train_file)
            input_nor_bias_train, output_hydration_nor_train, output_oxygen_nor_train, _ = data_preparation(file_path_test+train_file)
            input_nor_bias_all_train.append(input_nor_bias_train)
            output_hydration_nor_all_train.append(output_hydration_nor_train)
            output_oxygen_nor_all_train.append(output_oxygen_nor_train)

        input_nor_bias_all_train = np.vstack((input_nor_bias_all_train))
        output_hydration_nor_all_train = np.vstack((output_hydration_nor_all_train))
        output_oxygen_nor_all_train = np.vstack((output_oxygen_nor_all_train))

        #model_name = 'xgboostRegModel'
        model_fit_hydration = model_fitting(modelSelection(model_name), input_nor_bias_all_train, output_hydration_nor_all_train)
        prediction_hydration = model_fit_hydration.predict(input_nor_bias_test)
        target_data_hydration = output_hydration_nor_test
        error_std_rms_hydration, error_mean_rms_hydration = Error_estimation(target_data_hydration, prediction_hydration)
        error_mean_rms_hydration_all.append(error_mean_rms_hydration)
        error_std_rms_hydration_all.append(error_std_rms_hydration)

        model_fit_oxygen = model_fitting(modelSelection(model_name), input_nor_bias_all_train, output_oxygen_nor_all_train)
        prediction_oxygen = model_fit_oxygen.predict(input_nor_bias_test)
        target_data_oxygen = output_oxygen_nor_test
        error_std_rms_oxygen, error_mean_rms_oxygen = Error_estimation(target_data_oxygen, prediction_oxygen)
        error_mean_rms_oxygen_all.append(error_mean_rms_oxygen)
        error_std_rms_oxygen_all.append(error_std_rms_oxygen)
        #print("error_square_mean_oxygen_all = ", error_square_mean_oxygen_all)
        #print("error_square_std_oxygen_all = ", error_square_std_oxygen_all)
        #input("--------------Round end--------------")

        date_user_data_generated_header = ['Date', 'Temperature', 'Humidity', 'Avg3_Hydration', 'Avg3_Skinhealth', 'Skincare_Ratio', 'Age', 'Pred_Avg3_Hydration', 'Pred_Avg3_Skinhealth']

        data_generated = {
            date_user_data_generated_header[0]: np.asarray(df['Date'])[1:].flatten(),
            date_user_data_generated_header[1]: np.asarray(df['Temperature'])[1:].flatten(),
            date_user_data_generated_header[2]: np.asarray(df['Humidity'])[1:].flatten(),
            date_user_data_generated_header[3]: np.asarray(df['Avg3_Hydration'])[1:].flatten(),
            date_user_data_generated_header[4]: np.asarray(df['Avg3_Skinhealth'])[1:].flatten(),
            date_user_data_generated_header[5]: np.asarray(df['Skincare_Ratio'])[1:].flatten(),
            date_user_data_generated_header[6]: np.asarray(df['Age'])[1:].flatten(),
            date_user_data_generated_header[7]: prediction_hydration.flatten()*100,
            date_user_data_generated_header[8]: prediction_oxygen.flatten()*100,
        }

        date_user_data_generated_df = pd.DataFrame(data_generated)
        file_parent_dir = os.path.split(file)[0]
        file_name = os.path.split(file)[-1]
        date_user_data_generated_df.to_excel(file_parent_dir + '/Generated_User_Data/' + file_name, index=None)

    error_mean_rms_hydration_all_mean = np.mean(error_mean_rms_hydration_all)
    error_std_rms_hydration_all_mean = np.mean(error_std_rms_hydration_all)
    error_mean_rms_oxygen_all_mean = np.mean(error_mean_rms_oxygen_all)
    error_std_rms_oxygen_all_mean = np.mean(error_std_rms_oxygen_all)

    return error_mean_rms_hydration_all_mean, error_std_rms_hydration_all_mean, error_mean_rms_oxygen_all_mean, error_std_rms_oxygen_all_mean

def trainedModel(file_list, model_name, stored_path):
    input_nor_bias_all_train = []
    output_hydration_nor_all_train = []
    output_oxygen_nor_all_train = []
    for file in file_list:
        print("train_file = ", file)
        input_nor_bias_train, output_hydration_nor_train, output_oxygen_nor_train, df = data_preparation(file)

        input_nor_bias_all_train.append(input_nor_bias_train)
        output_hydration_nor_all_train.append(output_hydration_nor_train)
        output_oxygen_nor_all_train.append(output_oxygen_nor_train)

    input_nor_bias_all_train = np.vstack((input_nor_bias_all_train))
    output_hydration_nor_all_train = np.vstack((output_hydration_nor_all_train))

    output_oxygen_nor_all_train = np.vstack((output_oxygen_nor_all_train))

    # model_name = 'xgboostRegModel'
    model_fit_hydration = model_fitting(modelSelection(model_name), input_nor_bias_all_train, output_hydration_nor_all_train)
    model_fit_oxygen = model_fitting(modelSelection(model_name), input_nor_bias_all_train, output_oxygen_nor_all_train)

    with open(stored_path+str(model_name)+'_hydration'+'.pickle', 'wb') as hydration_model:
        pickle.dump(model_fit_hydration, hydration_model)
    with open(stored_path+str(model_name)+'_oxygen'+'.pickle', 'wb') as oxygen_model:
        pickle.dump(model_fit_oxygen, oxygen_model)

    return model_fit_hydration, model_fit_oxygen, input_nor_bias_all_train

def listdir():
    headpath = os.path.abspath('FA_main.py')[:-len('FA_main.py')]
    file_path = '/Use_Skincare/'
    path = headpath+file_path
    list_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        #print("os.path.splitext(file_path) = ", os.path.splitext(file_path))
        if os.path.isdir(file_path):
            pass
        elif os.path.splitext(file_path)[1] == '.xlsx':
            s = os.path.splitext(file_path)[0]
            if s.split('/')[-1][0] == 'P':
                list_name.append(file_path)
            else:
                continue
        else:
            continue
    #print(list_name)
    return list_name

def loadModel(model_name):
    with open(model_name, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def shapAnalysis(model_name, features_list, features_index_list, input_data_file_list, function_choice):
    model = loadModel(model_name)
    #features_list = ['Date', 'Temperature_current', 'Humidity_current',  'Temperature_next', 'Humidity_next',
    #                 'Avg_hydration', 'Avg_oxygen', 'Use_skincare', 'Bias']
    print("features_list = ", features_list)
    explainer = shap.TreeExplainer(model)

    input_nor_bias_all_train = []
    #output_hydration_nor_all_train = []
    #output_oxygen_nor_all_train = []
    for file in input_data_file_list:
        print("train_file = ", file)
        input_nor_bias_train, output_hydration_nor_train, output_oxygen_nor_train, df = data_preparation(file)

        input_nor_bias_all_train.append(input_nor_bias_train)
        #output_hydration_nor_all_train.append(output_hydration_nor_train)
        #output_oxygen_nor_all_train.append(output_oxygen_nor_train)

    input_nor_bias_all_train = np.vstack((input_nor_bias_all_train))
    #output_hydration_nor_all_train = np.vstack((output_hydration_nor_all_train))
    #output_oxygen_nor_all_train = np.vstack((output_oxygen_nor_all_train))
    #shap_data = np.vstack((features_list, input_nor_bias_all_train))
    shap_values_nor = explainer.shap_values(input_nor_bias_all_train)
    if function_choice == 'Overall features analysis':
        shap.summary_plot(shap_values_nor*100, input_nor_bias_all_train*100, feature_names=features_list)
    elif function_choice == 'Cross feature analysis':
        shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(input_nor_bias_all_train)
        shap.summary_plot(shap_interaction_values*100, input_nor_bias_all_train*100, max_display=input_nor_bias_all_train.shape[1], feature_names=np.asarray(features_list))
    elif function_choice == 'Two-Feature analysis':
        print("str(features_list[features_index_list[0]]) = ", str(features_list[features_index_list[0]]))
        print("str(features_list[features_index_list[1]]) = ", str(features_list[features_index_list[1]]))
        shap.dependence_plot(features_index_list[0], shap_values_nor*100, input_nor_bias_all_train*100, feature_names=features_list, interaction_index=str(features_list[features_index_list[1]]), show=False)
        #plt.title("Age dependence plot")
        plt.xlabel(str(features_list[features_index_list[0]]))
        plt.ylabel("SHAP value for the "+str(features_list[features_index_list[0]])+" feature")
        plt.show()
        shap.dependence_plot(features_index_list[1], shap_values_nor*100, input_nor_bias_all_train*100, feature_names=features_list, interaction_index=str(features_list[features_index_list[0]]), show=False)
        plt.xlabel(str(features_list[features_index_list[1]]))
        plt.ylabel("SHAP value for the " + str(features_list[features_index_list[1]]) + " feature")
        plt.show()
    else:
        pass


