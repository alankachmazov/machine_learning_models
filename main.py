import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time
import random
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# FURTHER IN THE CODE "b" STANDS FOR BIG (HIGH-DIMENSIONAL) AND "s" FOR SMALL (LOW-DIMENSIONAL)



# file paths to datasets. PLEASE PROVIDE THE CORRECT PATHS
filepath_s = "/internet_service_churn.csv"
filepath_b = "/Telco_customer_churn_cleaned.csv"



# DATA PREPOCESSING

# Low-Dimensional Dataset
small_df = pd.read_csv(filepath_s)
small_df = small_df.dropna()
small_df = small_df.drop("id", axis=1, inplace=False)
small_df.rename(columns={'reamining_contract': 'remaining_contract'}, inplace=True) # dataset contained a typo
# Scaling numerical variables of small dataset
numerical_variables_s = ["subscription_age", "bill_avg", "remaining_contract", "service_failure_count",
                       "download_avg", "upload_avg", "download_over_limit"]

# High-Dimensional Dataset
big_df = pd.read_csv(filepath_b)
big_df_initial = big_df
column_names = big_df.columns.tolist()

big_df  = big_df.drop("Churn Reason", axis = 1) # unnecessary
big_df  = big_df.drop("Offer", axis = 1)
big_df  = big_df.drop("Churn Category", axis = 1) # unnecessary
big_df  = big_df.drop("Lat Long", axis = 1)
big_df  = big_df.drop("Latitude", axis = 1)
big_df  = big_df.drop("Longitude", axis = 1)
big_df  = big_df.drop("Zip Code", axis = 1)
big_df  = big_df.drop("City", axis = 1)
big_df  = big_df.drop("State", axis = 1)
big_df  = big_df.drop("Country", axis = 1)
big_df  = big_df.drop("churn_rate", axis = 1) # duplicate of "Churn"

# Hot-key encoding
internet = pd.get_dummies(big_df["InternetService"])
internet.rename(columns={"No" : "InernetService"}, inplace=True)
big_df = pd.concat([big_df, internet], axis=1)
big_df = big_df.drop("InternetService", axis= 1)

payment = pd.get_dummies(big_df["PaymentMethod"])
big_df = pd.concat([big_df, payment], axis=1)
big_df = big_df.drop("PaymentMethod", axis= 1)

contract = pd.get_dummies(big_df["Contract"], prefix="Contract")
big_df = pd.concat([big_df, contract], axis=1)
big_df = big_df.drop("Contract", axis= 1)

# Remove unneeded variables
big_df = big_df.drop("customerID", axis = 1)
big_df = big_df.drop("Customer Status", axis = 1)

big_df = big_df.dropna()
column_names = big_df.columns.tolist()
big_df.rename(columns={"Churn" : "churn"}, inplace=True)

# Encoding Categorical variables to have 0 and 1 values
big_df['Unlimited Data'] = big_df['Unlimited Data'].replace({'Yes': 1, 'No': 0})
big_df['Premium Tech Support'] = big_df['Premium Tech Support'].replace({'Yes': 1, 'No': 0})
big_df['Streaming Music'] = big_df['Streaming Music'].replace({'Yes': 1, 'No': 0})
big_df['Referred a Friend'] = big_df['Referred a Friend'].replace({'Yes': 1, 'No': 0})
big_df['Married'] = big_df['Married'].replace({'Yes': 1, 'No': 0})
big_df['Under 30'] = big_df['Under 30'].replace({'Yes': 1, 'No': 0})
big_df['churn'] = big_df['churn'].replace({'Yes': 1, 'No': 0})
big_df['PaperlessBilling'] = big_df['PaperlessBilling'].replace({'Yes': 1, 'No': 0})
big_df['StreamingTV'] = big_df['StreamingTV'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
big_df['StreamingMovies'] = big_df['StreamingMovies'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
big_df['gender'] = big_df['gender'].replace({'Male': 1, 'Female': 0})
big_df['Partner'] = big_df['Partner'].replace({'Yes': 1, 'No': 0})
big_df['Dependents'] = big_df['Dependents'].replace({'Yes': 1, 'No': 0})
big_df['PhoneService'] = big_df['PhoneService'].replace({'Yes': 1, 'No': 0})
big_df['MultipleLines'] = big_df['MultipleLines'].replace({'Yes': 1, 'No': 0, 'No phone service': 0})
big_df['OnlineSecurity'] = big_df['OnlineSecurity'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
big_df['OnlineBackup'] = big_df['OnlineBackup'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
big_df['DeviceProtection'] = big_df['DeviceProtection'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
big_df['TechSupport'] = big_df['TechSupport'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
# Most of the models can handle Boolean values however not the ANN and SVM
big_df['DSL'] = big_df['DSL'].replace({True: 1, False: 0})
big_df['Fiber optic'] = big_df['Fiber optic'].replace({True: 1, False: 0})
big_df['InernetService'] = big_df['InernetService'].replace({True: 1, False: 0})
big_df['Bank transfer (automatic)'] = big_df['Bank transfer (automatic)'].replace({True: 1, False: 0})
big_df['Credit card (automatic)'] = big_df['Credit card (automatic)'].replace({True: 1, False: 0})
big_df['Electronic check'] = big_df['Electronic check'].replace({True: 1, False: 0})
big_df['Mailed check'] = big_df['Mailed check'].replace({True: 1, False: 0})
big_df['Contract_Month-to-month'] = big_df['Contract_Month-to-month'].replace({True: 1, False: 0})
big_df['Contract_One year'] = big_df['Contract_One year'].replace({True: 1, False: 0})
big_df['Contract_Two year'] = big_df['Contract_Two year'].replace({True: 1, False: 0})
# Scaling numerical columns for high-dimensional dataset
numerical_variables_b = ["Total Refunds", "Avg Monthly Long Distance Charges", "Avg Monthly GB Download", "Age", "CLTV",
                     "Count", "TotalCharges", "MonthlyCharges", "Total Revenue", "Total Long Distance Charges",
                     "Total Extra Data Charges", "tenure", "Satisfaction Score"]
scaler = StandardScaler()








############## THIS CODE IS NEEDED IF YOU WANT TO USE LOW-DIMENSIONAL DATASET ##############
data_scaled = small_df
data_scaled[numerical_variables_s] = scaler.fit_transform(data_scaled[numerical_variables_s])
#############################################################################################

############# COMMENT OUT CODE BELOW IF YOU WANT TO USE LOW-DIMENSIONAL DATASET #############
data_scaled = big_df
data_scaled[numerical_variables_b] = scaler.fit_transform(data_scaled[numerical_variables_b])
data_scaled = data_scaled.drop("Satisfaction Score", axis =1)
#############################################################################################







X = data_scaled.drop('churn', axis=1).values # remove churn variable
y = data_scaled['churn'].values # set target variable to be churn
data_scaled = data_scaled.drop('churn', axis=1)


# FEATURE IMPORTANCE ANALYSIS

def pca(data, n, cum_var): # n is the desired number of components & cum_var is desired cumulative variance explained
    pca = PCA(n_components=n)
    pca.fit(data)
    # transform data to its principal components
    transformed_data = pca.transform(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    # number of components required to explain specified variance. 1 is added because indexing starts from 0
    n_components = np.argmax(cumulative_explained_variance >= cum_var) + 1
    print(f"number of components that covers {cum_var * 100}% of variance is {n_components}")

    # plotting principal components
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()
    return transformed_data


def random_forest_reduction(X, y, importance_level): # specify desired cumulative importance level (from 0 to 1)
    rf_model = RandomForestClassifier()
    rf_model.fit(X, y)
    feature_importances = rf_model.feature_importances_

    # plotting
    feature_names = data_scaled.columns
    sorted_indices = np.argsort(feature_importances)[::1]
    feature_importances = feature_importances[sorted_indices]
    feature_names = feature_names[sorted_indices]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importances)), feature_importances, align='center')
    plt.yticks(range(len(feature_importances)), feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance of Random Forest Classifier')
    plt.show()

    # sort features in ascending order
    feature_ranks = np.argsort(feature_importances)
    ranked_feature_names = [feature_names[index] for index in feature_ranks]
    # revert the list to make it a in descending order
    ranked_feature_names = ranked_feature_names[::-1]
    print(ranked_feature_names)
    feature_importances = sorted(feature_importances, reverse=True)
    print(feature_importances)
    cumulative_importance = 0
    num_features_selected = 0
    # number of features that covers specified level of importance
    for idx, importance in enumerate(feature_importances):
        cumulative_importance += importance
        num_features_selected += 1
        if cumulative_importance >= importance_level:
            break

    # get the indices of the most important features
    print(num_features_selected)
    important_features = ranked_feature_names[0:num_features_selected]
    print(f"random forest considers important features to be: {important_features}")


def rfe(X, y): # use for estimator either LogisticRegression() or DecisionTreeClassifier()
    rfe = RFE(estimator=LogisticRegression(), n_features_to_select=2)
    rfe.fit(X, y)
    X = rfe.transform(X)
    model = DecisionTreeClassifier()
    pipeline = Pipeline(steps=[('s',rfe),('m',model)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # find which features are most important
    feature_rankings = rfe.ranking_
    # create an index array to represent feature indices
    feature_indices = np.arange(0, len(feature_rankings))
    # create a dictionary to associate feature indices with their rankings
    feature_rankings_dict = dict(zip(feature_indices, feature_rankings))
    # sort the features by their rankings to find the most important ones
    sorted_features = sorted(feature_rankings_dict.items(), key=lambda x: x[1]) #list of lists created from dictionary
    sorted_features_list = [i[0] for i in sorted_features] # list of indices of features in descending order
    features_names_list = data_scaled.columns[sorted_features_list]
    print("Most important features:")
    for feature_index, rank in sorted_features:
        print(f"Feature Index: {feature_index}, Rank: {rank}")
    return sorted_features_list, features_names_list



# PREDICTION MODEL BUILDING

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def decision_tree():
    clf = DecisionTreeClassifier()
    start_time = time.time()
    clf = clf.fit(X_train,y_train)
    training_time = time.time() - start_time
    start_time = time.time()
    y_pred = clf.predict(X_test)
    prediction_time = time.time() - start_time
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    execution_time = training_time + prediction_time
    print(f"Decision tree execution time: {execution_time} seconds")
    print(f"Decision tree training time: {training_time} seconds")
    print(f"Decision tree prediction time: {prediction_time} seconds")
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Decision Tree Precision: {precision * 100:.2f}%")
    print(f"Decision Tree Recall: {recall * 100:.2f}%")
    print(f"Decision Tree F1-Score: {f1 * 100:.2f}%")
    print(f"Decision Tree AUC-ROC: {auc_roc * 100:.2f}%")
    print(f"Decision Tree MCC: {mcc * 100:.2f}%")
    print(conf_matrix)

def random_forest():
    start_time = time.time()
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    start_time = time.time()
    y_pred = rf_model.predict(X_test)
    prediction_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    y_prob = rf_model.predict_proba(X_test)[:, 1] # probability of the positive class
    auc_roc = roc_auc_score(y_test, y_prob)
    execution_time = training_time + prediction_time
    print(f"Random forest execution time: {execution_time} seconds")
    print(f"Random forest training time: {training_time} seconds")
    print(f"Random forest prediction time: {prediction_time} seconds")
    print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
    print(f"Random Forest Precision: {precision * 100:.2f}%")
    print(f"Random Forest Recall: {recall * 100:.2f}%")
    print(f"Random Forest F1-Score: {f1 * 100:.2f}%")
    print(f"Random Forest AUC-ROC: {auc_roc * 100:.2f}%")
    print(f"Random Forest MCC: {mcc * 100:.2f}%")

def xgboost():
    xgb_classifier = xgb.XGBClassifier(n_estimators=500, objective='binary:logistic', tree_method='hist', eta=0.01, max_depth=10)
    start_time = time.time()
    xgb_classifier.fit(X_train, y_train)
    training_time = time.time() - start_time
    start_time = time.time()
    y_pred = xgb_classifier.predict(X_test)
    prediction_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_prob = xgb_classifier.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    execution_time = training_time + prediction_time
    print(f"XGBoost execution time: {execution_time} seconds")
    print(f"XGBoost training time: {training_time} seconds")
    print(f"XGBoost prediction time: {prediction_time} seconds")
    print(f"XGBoost Accuracy: {accuracy * 100:.2f}%")
    print(f"XGBoost Precision: {precision * 100:.2f}%")
    print(f"XGBoost Recall: {recall * 100:.2f}%")
    print(f"XGBoost F1-Score: {f1 * 100:.2f}%")
    print(f"XGBoost AUC-ROC: {auc_roc * 100:.2f}%")
    print(f"XGBoost MCC: {mcc * 100:.2f}%")

def logistic_regression():
    start_time = time.time()
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    start_time = time.time()
    y_pred = clf.predict(X_test)
    prediction_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    execution_time = training_time + prediction_time
    print(f"Logistic Regression execution time: {execution_time} seconds")
    print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")
    print(f"Logistic Regression Precision: {precision * 100:.2f}%")
    print(f"Logistic Regression Recall: {recall * 100:.2f}%")
    print(f"Logistic Regression F1-Score: {f1 * 100:.2f}%")
    print(f"Logistic Regression AUC-ROC: {auc_roc * 100:.2f}%")
    print(f"Logistic Regression MCC: {mcc * 100:.2f}%")

def naive_bayes():
    start_time = time.time()
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    start_time = time.time()
    y_pred = clf.predict(X_test)
    prediction_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    execution_time = training_time + prediction_time
    print(f"Naive Bayes execution time: {execution_time} seconds")
    print("Confusion Matrix:")
    print(cm)
    print(f'Naive Bayes Accuracy: {accuracy * 100:.2f}%')
    print(f"Naive Bayes Precision: {precision * 100:.2f}%")
    print(f"Naive Bayes Recall: {recall * 100:.2f}%")
    print(f"Naive Bayes F1-Score: {f1 * 100:.2f}%")
    print(f"Naive Bayes AUC-ROC: {auc_roc * 100:.2f}%")
    print(f"Naive Bayes MCC: {mcc * 100:.2f}%")

def knn(X_train, X_test, y_train, y_test):
    X_train = np.ascontiguousarray(X_train)
    X_test = np.ascontiguousarray(X_test)
    y_train = np.ascontiguousarray(y_train)
    y_test = np.ascontiguousarray(y_test)
    clf = KNeighborsClassifier(n_neighbors=11)
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    start_time = time.time()
    y_pred = clf.predict(X_test)
    prediction_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    execution_time = training_time + prediction_time
    print(f"KNN execution time: {execution_time} seconds")
    print(f'KNN Accuracy: {accuracy * 100:.2f}%')
    print(f"KNN Precision: {precision * 100:.2f}%")
    print(f"KNN Recall: {recall * 100:.2f}%")
    print(f"KNN F1-Score: {f1 * 100:.2f}%")
    print(f"KNN AUC-ROC: {auc_roc * 100:.2f}%")
    print(f"KNN MCC: {mcc * 100:.2f}%")

def ann(X_train, X_test, y_train, y_test, n1=32, n2=16):
    X_train = torch.FloatTensor(np.array(X_train))
    X_test = torch.FloatTensor(np.array(X_test))
    y_train = torch.FloatTensor(np.array(y_train))
    y_test = torch.FloatTensor(np.array(y_test))

    class ChurnPredictionModel(nn.Module):
        def __init__(self, input_dim):
            super(ChurnPredictionModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, n1)
            self.fc2 = nn.Linear(n1, n2)
            self.fc3 = nn.Linear(n2, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return self.sigmoid(x)

    input_dim = X_train.shape[1]
    model = ChurnPredictionModel(input_dim)
    criterion = nn.BCELoss()  # binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    num_epochs = 100

    for epoch in range(num_epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    training_time = time.time() - start_time
    start_time = time.time()

    with torch.no_grad():
        model.eval()
        y_pred = model(X_test)
        # convert predictions to binary values (0 or 1)
        y_pred = (y_pred > 0.5).float()

    prediction_time = time.time() - start_time

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    execution_time = training_time + prediction_time

    print(f"ANN execution time: {execution_time} seconds")
    print(f"ANN training time: {training_time} seconds")
    print(f"ANN prediction time: {prediction_time} seconds")
    print(f'ANN Accuracy: {accuracy * 100:.2f}%')
    print(f'ANN Precision: {precision * 100:.2f}%')
    print(f'ANN Recall: {recall * 100:.2f}%')
    print(f'ANN F1 Score: {f1 * 100:.2f}%')
    print(f'ANN AUC-ROC: {auc_roc * 100:.2f}%')
    print(f"ANN MCC: {mcc * 100:.2f}%")

def svm(X_train, X_test, y_train, y_test):
    X_train = torch.FloatTensor(np.array(X_train))
    X_test = torch.FloatTensor(np.array(X_test))
    y_train = torch.FloatTensor(np.array(y_train))
    y_test = torch.FloatTensor(np.array(y_test))
    class SVM(nn.Module):
        def __init__(self, input_dim):
            super(SVM, self).__init__()
            self.fc = nn.Linear(input_dim, 1)

        def forward(self, x):
            return self.fc(x)

    input_dim = X_train.size(1)
    svm_model = SVM(input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(svm_model.parameters(), lr=0.01)
    start_time = time.time()
    # training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = svm_model(X_train).squeeze()
        loss = criterion(output, y_train.float())
        loss.backward()  # backpropagation
        optimizer.step()  # updating model parameters
    training_time = time.time() - start_time
    # evaluation
    start_time = time.time()
    with torch.no_grad():

        test_output = svm_model(X_test).squeeze()
        predicted_labels = (test_output > 0.0).float()  # binary classification
        prediction_time = time.time() - start_time
        accuracy = accuracy_score(y_test, predicted_labels)
        precision = precision_score(y_test, predicted_labels)
        recall = recall_score(y_test, predicted_labels)
        f1 = f1_score(y_test, predicted_labels)
        auc_roc = roc_auc_score(y_test, test_output.sigmoid().numpy())
        mcc = matthews_corrcoef(y_test, predicted_labels)
        execution_time = training_time + prediction_time
        print(f"SVM execution time: {execution_time} seconds")
        print(f"SVM training time: {training_time}")
        print(f"SVM prediction time: {prediction_time}")
        print(f"SVM Accuracy: {accuracy * 100:.2f}%")
        print(f"SVM Precision: {precision * 100:.2f}%")
        print(f"SVM Recall: {recall * 100:.2f}%")
        print(f"SVM F1-Score: {f1 * 100:.2f}%")
        print(f"SVM AUC-ROC: {auc_roc * 100:.2f}%")
        print(f"SVM MCC: {mcc * 100:.2f}%")


######################################## GWO ######################################################
# objective function that needs to be optimized to find best number of neurons for ANN layers
# d1 and d2 are the number of neurons on 1st and 2nd layer of ANN

X_train_gwo = torch.FloatTensor(X_train)
X_test_gwo = torch.FloatTensor(X_test)
y_train_gwo = torch.FloatTensor(y_train)
y_test_gwo = torch.FloatTensor(y_test)
def objective_function(d1, d2, X_train_gwo=X_train_gwo, X_test_gwo=X_test_gwo, y_train_gwo=y_train_gwo, y_test_gwo=y_test_gwo):
    class ChurnPredictionModel(nn.Module):
        def __init__(self, input_dim):
            super(ChurnPredictionModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, int(d1))
            self.fc2 = nn.Linear(int(d1), int(d2))
            self.fc3 = nn.Linear(int(d2), 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return self.sigmoid(x)

    input_dim = X_train_gwo.shape[1]
    model = ChurnPredictionModel(input_dim)

    criterion = nn.BCELoss()  # binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100

    for epoch in range(num_epochs):
        outputs = model(X_train_gwo)
        loss = criterion(outputs, y_train_gwo.view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        y_pred = model(X_test_gwo)
        y_pred = (y_pred > 0.5).float()

    return y_test_gwo, y_pred

def fitness_function(d1, d2):
    y_test, y_pred = objective_function(d1, d2)
    recall = recall_score(y_test, y_pred)
    return recall

# we use fitness function, number of wolves and iterations, and number of neurons range for each layer as
# parameters for grey wolf optimization function
def grey_wolf(fitness_function, num_wolves, num_iterations, d1_min, d1_max, d2_min, d2_max):
    wolf_positions1 = [random.randint(d1_min, d1_max) for i in range(num_wolves)]
    wolf_positions2 = [random.randint(d2_min, d2_max) for i in range(num_wolves)]
    # sort wolves by the fitness function and turn them back into lists from tuples
    wolf_positions1, wolf_positions2 = zip(*sorted(
        zip(wolf_positions1, wolf_positions2),
        key=lambda x: fitness_function(x[0], x[1])
    ))
    wolf_positions1 = list(wolf_positions1)
    wolf_positions2 = list(wolf_positions2)

    for iteration in range(num_iterations):
        alpha = 2.0 - (2.0 / num_iterations) * iteration  # linearly decrease over the course of iterations
        for i in range(num_wolves):
            a = 2.0 * alpha * random.random() - alpha
            # add randomness
            c1 = random.random() * 2
            c2 = random.random() * 2
            c3 = random.random() * 2
            # calculating distances
            d_alpha_x = abs(c1 * wolf_positions1[0] - wolf_positions1[i])  # distance between wolves
            d_alpha_y = abs(c1 * wolf_positions2[0] - wolf_positions2[i])
            # calculating possible positions
            x1_x = max(d1_min, min(wolf_positions1[0] - a * d_alpha_x, d1_max))  # x stands for first layer
            x1_y = max(d2_min, min(wolf_positions2[0] - a * d_alpha_y, d2_max))  # y stands for second layer

            d_beta_x = abs(c2 * wolf_positions1[1] - wolf_positions1[i])
            d_beta_y = abs(c2 * wolf_positions2[1] - wolf_positions2[i])

            x2_x = max(d1_min, min(wolf_positions1[1] - a * d_beta_x, d1_max))
            x2_y = max(d2_min, min(wolf_positions2[0] - a * d_beta_y, d2_max))
            d_delta_x = abs(c3 * wolf_positions1[2] - wolf_positions1[i])
            d_delta_y = abs(c3 * wolf_positions2[2] - wolf_positions2[i])

            x3_x = max(d1_min, min(wolf_positions1[2] - a * d_delta_x, d1_max))
            x3_y = max(d2_min, min(wolf_positions2[0] - a * d_delta_y, d2_max))

            # update the position of the current wolf
            new_position1 = (x1_x + x2_x + x3_x) / 3.0
            new_position2 = (x1_y + x2_y + x3_y) / 3.0

            # evaluate fitness of the new position
            new_fitness = fitness_function(new_position1, new_position2)

            # update the wolf position if the new position is better
            if new_fitness > fitness_function(wolf_positions1[i], wolf_positions2[i]):
                wolf_positions1[i] = int(new_position1)
                wolf_positions2[i] = int(new_position2)


        wolf_positions1, wolf_positions2 = zip(*sorted(
            zip(wolf_positions1, wolf_positions2),
            key=lambda x: fitness_function(x[0], x[1])
        ))
        wolf_positions1 = list(wolf_positions1)
        wolf_positions2 = list(wolf_positions2)
        print(f"wolves positions for first layer are: {wolf_positions1}")
        print(f"wolves positions for second layer are: {wolf_positions2}")
    best_d1 = wolf_positions1[0]
    best_d2 = wolf_positions2[0]
    print(f"Best number of neurons on first layer: {best_d1},Best number of neurons on second layer: {best_d2}")

d1_min = 32
d1_max = 64
d2_min = 10
d2_max = 31

#grey_wolf(fitness_function, 12, 10, d1_min, d1_max, d2_min, d2_max)


# PLOTTING
# correlation between total revenue and monthly charges
def cor_plot1():
    plt.scatter(big_df_initial["Total Revenue"], big_df_initial["MonthlyCharges"], marker='o', s=5, c='#2596be', label='Data Points')
    plt.xlabel("Total Revenue")
    plt.ylabel("Monthly Charges")
    plt.show()

# plot to show the influence of artificial variable "Satisfaction Score"
def yes_churn_plot():
    yes_churn = big_df_initial[big_df_initial["Churn"] == "Yes"]
    plt.bar(yes_churn["Churn"], yes_churn["Satisfaction Score"])
    plt.xlabel("Satisfaction Score")
    plt.ylabel("Number of Churners")
    yes_score = {
        "1" : [(yes_churn["Satisfaction Score"] == 1).sum()],
        "2" : [(yes_churn["Satisfaction Score"] == 2).sum()],
        "3" : [(yes_churn["Satisfaction Score"] == 3).sum()],
        "4" : [(yes_churn["Satisfaction Score"] == 4).sum()],
        "5" : [(yes_churn["Satisfaction Score"] == 5).sum()]
    }
    yes_df = pd.DataFrame(yes_score)
    colors = ["#9CDBF5", "#F5E39C", "#9CA6F5", "#F59CA8", "#F5A29C"]
    for column in yes_df.columns:
        plt.bar(column, yes_df[column], color = colors[int(column)-1])

    plt.xlim(0, 5.5)
    plt.show()

def no_churn_plot():
    no_churn = big_df_initial[big_df_initial["Churn"] == "No"]
    no_score = {
        "1" : [(no_churn["Satisfaction Score"] == 1).sum()],
        "2" : [(no_churn["Satisfaction Score"] == 2).sum()],
        "3" : [(no_churn["Satisfaction Score"] == 3).sum()],
        "4" : [(no_churn["Satisfaction Score"] == 4).sum()],
        "5" : [(no_churn["Satisfaction Score"] == 5).sum()]
    }
    no_df = pd.DataFrame(no_score)
    colors = ["#9CDBF5", "#F5E39C", "#9CA6F5", "#F59CA8", "#F5A29C"]

    for column in no_df.columns:
        plt.bar(column, no_df[column], color = colors[int(column)-1])
    plt.ylabel("Number of Non-Churners")
    plt.show()


# changes in ANN preformance based on threshold
def ann_threshold_plot():
    threshold_05 = {
        "Accuracy": 80.98,
        "Precision": 68.03,
        "Recall": 62.25,
        "F1 Score": 65.01,
        "AUC-ROC": 75.33
    }
    threshold_025 = {
        "Accuracy": 76.58,
        "Precision": 55.89,
        "Recall": 83.00,
        "F1 Score": 66.80,
        "AUC-ROC": 78.52
    }
    # calculate the improvement or decrease for each metric
    changes = {metric: threshold_025[metric] - threshold_05[metric] for metric in threshold_05}
    colors = ['red' if val < 0 else 'green' for val in changes.values()]
    initial_values = list(threshold_05.values())
    # plot the initial bars for 0.5 threshold
    plt.bar(threshold_05.keys(), initial_values, color='lightgray')
    # plot the changes for 0.25 threshold
    plt.bar(threshold_05.keys(), [abs(val) for val in changes.values()], bottom=initial_values, color=colors)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Improvement/Decrease in Metrics')
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CALLING FUNCTIONS

### BASIC models ###
print("BASIC")
#decision_tree()
#random_forest()
#xgboost()
#logistic_regression()
#naive_bayes()
#knn(X_train, X_test, y_train, y_test)
#svm(X_train, X_test, y_train, y_test)
#ann(X_train, X_test, y_train, y_test, 46, 11)
#ann(X_train, X_test, y_train, y_test, 59, 22)


### PCA models ###
print("PCA")

#data = pca(X, 12, 0.88) # 6 FOR LOW DIMENSIONAL, 12 FOR HIGH DIMENSIONAL
#X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

#decision_tree()
#random_forest()
#xgboost()
#logistic_regression()
#naive_bayes()
#knn(X_train, X_test, y_train, y_test)
#svm(X_train, X_test, y_train, y_test)
#ann(X_train, X_test, y_train, y_test, 46, 11)
#ann(X_train, X_test, y_train, y_test, 59, 22)


### RANDOM FOREST models ###
print("Random Forest")
#random_forest_reduction(X, y, 0.8)

# USE "data" BELOW FOR LOW DIMENSIONAL DATASET, OTHERWISE COMMENT IT OUT
#data = data_scaled[['remaining_contract', 'download_avg', 'upload_avg', 'subscription_age', 'bill_avg']]

# USE "data" BELOW FOR HIGH DIMENSIONAL DATASET, OTHERWISE COMMENT IT OUT
data = data_scaled[['MonthlyCharges', 'TotalCharges', 'Total Revenue', 'tenure', 'Number of Referrals', 'CLTV',
                    'Avg Monthly GB Download', 'Total Long Distance Charges', 'Contract_Month-to-month', 'Age',
                    'Avg Monthly Long Distance Charges', 'Electronic check', 'Fiber optic', 'Contract_Two year',
                    'PaperlessBilling', 'OnlineSecurity']]

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)
#decision_tree()
#random_forest()
#xgboost()
#logistic_regression()
#naive_bayes()
#knn(X_train, X_test, y_train, y_test)
#svm(X_train, X_test, y_train, y_test)
#ann(X_train, X_test, y_train, y_test, 46, 11)
#ann(X_train, X_test, y_train, y_test, 59, 22)


### RFE Models ###
#features, names = rfe(X,y)
#print(features)
#print(names)
print("RFE")

# USE "data" BELOW FOR LOW DIMENSIONAL DATASET, OTHERWISE COMMENT IT OUT
#data = data_scaled[['is_tv_subscriber', 'remaining_contract', 'download_avg', 'subscription_age', 'download_over_limit']]

# USE "data" BELOW FOR HIGH DIMENSIONAL DATASET, OTHERWISE COMMENT IT OUT
data = data_scaled[['Contract_Month-to-month', 'Contract_Two year', 'Fiber optic',
       'InernetService', 'tenure', 'TotalCharges', 'PhoneService',
       'Referred a Friend', 'Number of Referrals', 'OnlineSecurity',
       'TechSupport', 'Electronic check', 'PaperlessBilling', 'SeniorCitizen',
       'Streaming Music', 'StreamingTV']]

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

#decision_tree()
#random_forest()
#xgboost()
#logistic_regression()
#naive_bayes()
#knn(X_train, X_test, y_train, y_test)
#svm(X_train, X_test, y_train, y_test)
#ann(X_train, X_test, y_train, y_test, 46, 11)
#ann(X_train, X_test, y_train, y_test, 59, 22)


