import xgboost as xgb

def modelSelection(modelName):
    print("str(modelName) = ", str(modelName))
    print("str(modelName) == 'xgboostRegModel' = ", str(modelName) == 'xgboostRegModel')
    if str(modelName) == 'xgboostRegModel':
        XGBReg_model = xgb.XGBRegressor(n_estimators=100,learning_rate=0.1)
        return XGBReg_model
    else:
        print("No such model, please make sure the model exists!")

