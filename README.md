# FeatureAnalysisSystemGUI
In this project, you can try to use a intuition GUI interface to (1)choose a batch of data, (2)train a model, (3)run performance validation, and (4)run feature analysis in just one GUI, and you can also change some code to meet your application's needs. If you fimiliar with pyQt, it can also be built as a standalone application.

## Requirements
pandas
numpy
xgboost
sklearn
shap
matplotlib
pathlib
py2app

## Steps
Run AI_Test_Platform_and_Feature_Analysis_GUI.py first to pop the GUI
### Training and validation
1. Data Set Selection: Choose the batch of dataset.
2. Model Selection: You can add you model to "New_Models" directory first, then it can be browsed at this step.
3. Performance Validation: Here automatically run K-fold cross validation.
4. Model Training: After running Performance Validation and the accuracy is expected, then training the model(at step 3) with all dataset you choose(at step 1).
### Feature analysis (All shows you a figure)
1. Overall features analysis: shows you how each feature influence the target
2. Cross feature analysis: shows you cross influence between every 2 inputs
3. Two-Feature analysis: shows you the influence of 2 features(write in "Feature_Names" directory) that you choosed on target

#### For this project we facous on the facial related data(such as hydration, oxygen, etc.) and weather data(such as temperature and humidity), so you can try to modify the features to meet your requirement and the applications.
