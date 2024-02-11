import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.base import clone
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from pgmpy.models import BayesianNetwork
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



def k_fold(X, y, n_folds, classifier, verbose=False):
    kf = KFold(n_splits=n_folds, shuffle=True)
    f1 = 0  # F1 score
    accuracy = 0  # Accuracy
    precision = 0  # Precision
    recall = 0  # Recall
    j = 1  # contatore
    best_f1 = 0  # Migliore F1 score
    best_classifier = None  # Migliore classificatore

    for train_indexes, test_indexes in kf.split(X, y):
        curr_classifier = clone(classifier)
        curr_classifier = curr_classifier.fit(X.iloc[train_indexes], y[train_indexes])

        # Calcolo Accuracy, Precision, Recall, F1
        curr_accuracy = accuracy_score(y[test_indexes], curr_classifier.predict(X.iloc[test_indexes]))
        curr_precision = precision_score(y[test_indexes], curr_classifier.predict(X.iloc[test_indexes]),
                                         average='weighted', zero_division=1)
        curr_recall = recall_score(y[test_indexes], curr_classifier.predict(X.iloc[test_indexes]), average='weighted',
                                   zero_division=1)
        curr_f1 = f1_score(y[test_indexes], curr_classifier.predict(X.iloc[test_indexes]), average='weighted',
                           zero_division=1)

        if verbose:
            print("Fold " + str(j) + "/" + str(n_folds))
            print("F1: ", str(curr_f1))
            print("Accuracy: ", str(curr_accuracy))
            print("Precision: ", str(curr_precision))
            print("Recall: ", str(curr_recall))

        # Controlla se il classificatore attuale è migliore del miglior classificatore finora
        if curr_f1 > best_f1:
            best_classifier = curr_classifier
            best_f1 = curr_f1

        j += 1
        f1 += curr_f1
        accuracy += curr_accuracy
        precision += curr_precision
        recall += curr_recall

    mean_f1 = f1 / n_folds
    mean_accuracy = accuracy / n_folds
    mean_precision = precision / n_folds
    mean_recall = recall / n_folds

    return best_classifier, mean_f1, mean_accuracy, mean_precision, mean_recall




def print_metrics(y_true, y_pred, model_name):
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test F1 per {model_name}: {f1}")
    print(f"Test Precision per {model_name}: {precision}")
    print(f"Test Recall per {model_name}: {recall}")
    print(f"Test Accuracy per {model_name}: {accuracy}")

def print_feature_importances(model, X_columns, model_name):
    importances = model.feature_importances_
    features = pd.DataFrame({
        'Feature': X_columns,
        'Importance': importances
    })
    features.sort_values(by='Importance', ascending=False, inplace=True)
    print(f"Feature importances for {model_name}:\n", features)




def tune_model(X, y, model, param_grid):
    # Inizializza la ricerca su griglia
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

    # Esegui la ricerca su griglia
    grid_search.fit(X, y)

    # Stampa i migliori iperparametri
    print("Best parameters: ", grid_search.best_params_)

    # Restituisci il miglior modello
    return grid_search.best_estimator_

def param_rf():
    param_grid = {
        'max_depth': [None, 3, 5, 10, 15],
        'min_samples_split': [2, 5, 10, 20],
        'criterion': ['gini', 'entropy'],
        'n_estimators': [10, 20, 50, 100]
    }
    return param_grid

def param_dt():
    param_grid = {
        'max_depth': [None, 3, 5, 10, 15],
        'min_samples_split': [2, 5, 10, 20],
        'criterion': ['gini', 'entropy']
    }
    return param_grid

def param_ada():
    param_grid = {
        'n_estimators': [30, 50, 70, 100, 150],
        'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0]
    }
    return param_grid

def param_lgbm():
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 1.0],
        'max_depth': [3, 5],
        'num_leaves': [31, 60],
        'min_child_samples': [20, 40]
    }
    return param_grid




def plot_learning_curve(estimator, X, y, save_path=None, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.plot(train_sizes, train_scores_mean, '-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, '-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(f'../immagini/{save_path}.png', dpi=400)
    plt.clf()




def my_rf():
    # Carica i dati
    df = pd.read_csv("working_dataset2.csv")

    scaler = StandardScaler()

    # Prepara i dati per "PART 1-2"
    X1 = df[["NUM_CRIMES_TYPE", "NUM_CRIMES_CODE", "NUM_CRIMES_TYPE_DESC"]].copy()
    y1 = df["PART 1-2"]
    X1 = pd.DataFrame(scaler.fit_transform(X1), columns=X1.columns)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "PART 1-2"
    model1 = tune_model(X1_train, y1_train, RandomForestClassifier(), param_rf())
    best_model1, mean_f1_1, mean_accuracy1, mean_precision1, mean_recall1 = k_fold(X1_train.reset_index(drop=True),
      y1_train.reset_index(drop=True), 5, model1)
    print_metrics(y1_test, best_model1.predict(X1_test), "PART 1-2")
    print_feature_importances(best_model1, X1.columns, "PART 1-2")
    plot_learning_curve(best_model1, X1_train, y1_train,"RF_PART1-2")


    # Prepara i dati per "CRM CD"
    X2 = df[["NUM_CRIMES_CODE", "NUM_CRIMES_TYPE_DESC"]].copy()
    X2["PART 1-2"] = best_model1.predict(X1)
    y2 = df["CRM CD"]
    X2 = pd.DataFrame(scaler.fit_transform(X2), columns=X2.columns)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "CRM CD"
    model2 = tune_model(X2_train, y2_train, RandomForestClassifier(), param_rf())
    best_model2, mean_f1_2, mean_accuracy2, mean_precision2, mean_recall2 = k_fold(X2_train.reset_index(drop=True), y2_train.reset_index(drop=True), 5, model2)
    print_metrics(y2_test, best_model2.predict(X2_test), "CRM CD")
    print_feature_importances(best_model2, X2.columns, "CRM CD")
    plot_learning_curve(best_model2, X2_train, y2_train,"RF_CRM-CD")


    # Prepara i dati per "AREA"
    X3 = df[["NUM_CRIMES_AREA", "NUM_CRIMES_AREA_NAME", "NUM_CRIMES_DISTRICT"]].copy()
    y3 = df["AREA"]
    X3 = pd.DataFrame(scaler.fit_transform(X3), columns=X3.columns)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "AREA"
    model3 = tune_model(X3_train, y3_train, RandomForestClassifier(), param_rf())
    best_model3, mean_f1_3, mean_accuracy3, mean_precision3, mean_recall3 = k_fold(X3_train.reset_index(drop=True),
      y3_train.reset_index(drop=True), 5, model3)
    print_metrics(y3_test, best_model3.predict(X3_test), "AREA")
    print_feature_importances(best_model3, X3.columns, "AREA")
    plot_learning_curve(best_model3, X3_train, y3_train,"RF_AREA")


    # Prepara i dati per "RPT DIST NO"
    X4 = df[["NUM_CRIMES_DISTRICT", "NUM_CRIMES_DATE"]].copy()
    X4["AREA"] = best_model3.predict(X3)
    y4 = df["RPT DIST NO"]
    X4 = pd.DataFrame(scaler.fit_transform(X4), columns=X4.columns)
    X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "RPT DIST NO"
    model4 = tune_model(X4_train, y4_train, RandomForestClassifier(), param_rf())
    best_model4, mean_f1_4, mean_accuracy4, mean_precision4, mean_recall4 = k_fold(X4_train.reset_index(drop=True),
                                                                                   y4_train.reset_index(drop=True), 5,
                                                                                   model4)
    print_metrics(y4_test, best_model4.predict(X4_test), "RPT DIST NO")
    print_feature_importances(best_model4, X4.columns, "RPT DIST NO")
    plot_learning_curve(best_model4, X4_train, y4_train,"RF_RPT-DIST-NO")


    # Prepara i dati per "PREMISE_CODE"
    X5 = df[["NUM_CRIMES_PREMISE_DESC", "NUM_CRIMES_PREMISE", "NUM_CRIMES_DISTRICT"]].copy()
    y5 = df["PREMISE_CODE"]
    X5 = pd.DataFrame(scaler.fit_transform(X5), columns=X5.columns)
    X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "PREMISE_CODE"
    model5 = tune_model(X5_train, y5_train, RandomForestClassifier(), param_rf())
    best_model5, mean_f1_5, mean_accuracy5, mean_precision5, mean_recall5 = k_fold(X5_train.reset_index(drop=True), y5_train.reset_index(drop=True), 5, model5)
    print_metrics(y5_test, best_model5.predict(X5_test), "PREMISE_CODE")
    print_feature_importances(best_model5, X5.columns, "PREMISE_CODE")
    plot_learning_curve(best_model5, X5_train, y5_train,"RF_PREMISE-CODE")



def my_dt():
    # Carica i dati
    df = pd.read_csv("working_dataset2.csv")

    scaler = StandardScaler()

    # Prepara i dati per "PART 1-2"
    X1 = df[["NUM_CRIMES_TYPE", "NUM_CRIMES_CODE", "NUM_CRIMES_TYPE_DESC"]].copy()
    y1 = df["PART 1-2"]
    X1 = pd.DataFrame(scaler.fit_transform(X1), columns=X1.columns)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "PART 1-2"
    model1 = tune_model(X1_train, y1_train, DecisionTreeClassifier(), param_dt())
    best_model1, mean_f1_1, mean_accuracy1, mean_precision1, mean_recall1 = k_fold(X1_train.reset_index(drop=True),
      y1_train.reset_index(drop=True), 5, model1)
    print_metrics(y1_test, best_model1.predict(X1_test), "PART 1-2")
    print_feature_importances(best_model1, X1.columns, "PART 1-2")
    plot_learning_curve(best_model1, X1_train, y1_train,"DT_PART1-2")


    # Prepara i dati per "CRM CD"
    X2 = df[["NUM_CRIMES_CODE", "NUM_CRIMES_TYPE_DESC"]].copy()
    X2["PART 1-2"] = best_model1.predict(X1)
    y2 = df["CRM CD"]
    X2 = pd.DataFrame(scaler.fit_transform(X2), columns=X2.columns)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "CRM CD"
    model2 = tune_model(X2_train, y2_train, DecisionTreeClassifier(), param_dt())
    best_model2, mean_f1_2, mean_accuracy2, mean_precision2, mean_recall2 = k_fold(X2_train.reset_index(drop=True), y2_train.reset_index(drop=True), 5, model2)
    print_metrics(y2_test, best_model2.predict(X2_test), "CRM CD")
    print_feature_importances(best_model2, X2.columns, "CRM CD")
    plot_learning_curve(best_model2, X2_train, y2_train,"DT_CRM-CD")


    # Prepara i dati per "AREA"
    X3 = df[["NUM_CRIMES_AREA", "NUM_CRIMES_AREA_NAME", "NUM_CRIMES_DISTRICT"]].copy()
    y3 = df["AREA"]
    X3 = pd.DataFrame(scaler.fit_transform(X3), columns=X3.columns)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "AREA"
    model3 = tune_model(X3_train, y3_train, DecisionTreeClassifier(), param_dt())
    best_model3, mean_f1_3, mean_accuracy3, mean_precision3, mean_recall3 = k_fold(X3_train.reset_index(drop=True),
      y3_train.reset_index(drop=True), 5, model3)
    print_metrics(y3_test, best_model3.predict(X3_test), "AREA")
    print_feature_importances(best_model3, X3.columns, "AREA")
    plot_learning_curve(best_model3, X3_train, y3_train,"DT_AREA")


    # Prepara i dati per "RPT DIST NO"
    X4 = df[["NUM_CRIMES_DISTRICT", "NUM_CRIMES_DATE"]].copy()
    X4["AREA"] = best_model3.predict(X3)
    y4 = df["RPT DIST NO"]
    X4 = pd.DataFrame(scaler.fit_transform(X4), columns=X4.columns)
    X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "RPT DIST NO"
    model4 = tune_model(X4_train, y4_train, DecisionTreeClassifier(), param_dt())
    best_model4, mean_f1_4, mean_accuracy4, mean_precision4, mean_recall4 = k_fold(X4_train.reset_index(drop=True),
                                                                                   y4_train.reset_index(drop=True), 5,
                                                                                   model4)
    print_metrics(y4_test, best_model4.predict(X4_test), "RPT DIST NO")
    print_feature_importances(best_model4, X4.columns, "RPT DIST NO")
    plot_learning_curve(best_model4, X4_train, y4_train,"DT_RPT-DIST-NO")


    # Prepara i dati per "PREMISE_CODE"
    X5 = df[["NUM_CRIMES_PREMISE_DESC", "NUM_CRIMES_PREMISE", "NUM_CRIMES_DISTRICT"]].copy()
    y5 = df["PREMISE_CODE"]
    X5 = pd.DataFrame(scaler.fit_transform(X5), columns=X5.columns)
    X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "PREMISE_CODE"
    model5 = tune_model(X5_train, y5_train, DecisionTreeClassifier(), param_dt())
    best_model5, mean_f1_5, mean_accuracy5, mean_precision5, mean_recall5 = k_fold(X5_train.reset_index(drop=True), y5_train.reset_index(drop=True), 5, model5)
    print_metrics(y5_test, best_model5.predict(X5_test), "PREMISE_CODE")
    print_feature_importances(best_model5, X5.columns, "PREMISE_CODE")
    plot_learning_curve(best_model5, X5_train, y5_train,"DT_PREMISE-CODE")

def my_ada():
    # Carica i dati
    df = pd.read_csv("working_dataset2.csv")

    scaler = StandardScaler()

    # Prepara i dati per "PART 1-2"
    X1 = df[["NUM_CRIMES_TYPE", "NUM_CRIMES_CODE", "NUM_CRIMES_TYPE_DESC"]].copy()
    y1 = df["PART 1-2"]
    X1 = pd.DataFrame(scaler.fit_transform(X1), columns=X1.columns)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "PART 1-2"
    model1 = tune_model(X1_train, y1_train, AdaBoostClassifier(), param_ada())
    best_model1, mean_f1_1, mean_accuracy1, mean_precision1, mean_recall1 = k_fold(X1_train.reset_index(drop=True),
      y1_train.reset_index(drop=True), 5, model1)
    print_metrics(y1_test, best_model1.predict(X1_test), "PART 1-2")
    print_feature_importances(best_model1, X1.columns, "PART 1-2")
    plot_learning_curve(best_model1, X1_train, y1_train,"ADA_PART1-2")


    # Prepara i dati per "CRM CD"
    X2 = df[["NUM_CRIMES_CODE", "NUM_CRIMES_TYPE_DESC"]].copy()
    X2["PART 1-2"] = best_model1.predict(X1)
    y2 = df["CRM CD"]
    X2 = pd.DataFrame(scaler.fit_transform(X2), columns=X2.columns)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "CRM CD"
    model2 = tune_model(X2_train, y2_train, AdaBoostClassifier(), param_ada())
    best_model2, mean_f1_2, mean_accuracy2, mean_precision2, mean_recall2 = k_fold(X2_train.reset_index(drop=True), y2_train.reset_index(drop=True), 5, model2)
    print_metrics(y2_test, best_model2.predict(X2_test), "CRM CD")
    print_feature_importances(best_model2, X2.columns, "CRM CD")
    plot_learning_curve(best_model2, X2_train, y2_train,"ADA_CRM-CD")


    # Prepara i dati per "AREA"
    X3 = df[["NUM_CRIMES_AREA", "NUM_CRIMES_AREA_NAME", "NUM_CRIMES_DISTRICT"]].copy()
    y3 = df["AREA"]
    X3 = pd.DataFrame(scaler.fit_transform(X3), columns=X3.columns)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "AREA"
    model3 = tune_model(X3_train, y3_train, AdaBoostClassifier(), param_ada())
    best_model3, mean_f1_3, mean_accuracy3, mean_precision3, mean_recall3 = k_fold(X3_train.reset_index(drop=True),
      y3_train.reset_index(drop=True), 5, model3)
    print_metrics(y3_test, best_model3.predict(X3_test), "AREA")
    print_feature_importances(best_model3, X3.columns, "AREA")
    plot_learning_curve(best_model3, X3_train, y3_train,"ADA_AREA")


    # Prepara i dati per "RPT DIST NO"
    X4 = df[["NUM_CRIMES_DISTRICT", "NUM_CRIMES_DATE"]].copy()
    X4["AREA"] = best_model3.predict(X3)
    y4 = df["RPT DIST NO"]
    X4 = pd.DataFrame(scaler.fit_transform(X4), columns=X4.columns)
    X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "RPT DIST NO"
    model4 = tune_model(X4_train, y4_train, AdaBoostClassifier(), param_ada())
    best_model4, mean_f1_4, mean_accuracy4, mean_precision4, mean_recall4 = k_fold(X4_train.reset_index(drop=True),
                                                                                   y4_train.reset_index(drop=True), 5,
                                                                                   model4)
    print_metrics(y4_test, best_model4.predict(X4_test), "RPT DIST NO")
    print_feature_importances(best_model4, X4.columns, "RPT DIST NO")
    plot_learning_curve(best_model4, X4_train, y4_train,"ADA_RPT-DIST-NO")


    # Prepara i dati per "PREMISE_CODE"
    X5 = df[["NUM_CRIMES_PREMISE_DESC", "NUM_CRIMES_PREMISE", "NUM_CRIMES_DISTRICT"]].copy()
    y5 = df["PREMISE_CODE"]
    X5 = pd.DataFrame(scaler.fit_transform(X5), columns=X5.columns)
    X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "PREMISE_CODE"
    model5 = tune_model(X5_train, y5_train, AdaBoostClassifier(), param_ada())
    best_model5, mean_f1_5, mean_accuracy5, mean_precision5, mean_recall5 = k_fold(X5_train.reset_index(drop=True), y5_train.reset_index(drop=True), 5, model5)
    print_metrics(y5_test, best_model5.predict(X5_test), "PREMISE_CODE")
    print_feature_importances(best_model5, X5.columns, "PREMISE_CODE")
    plot_learning_curve(best_model5, X5_train, y5_train,"ADA_PREMISE-CODE")
def my_gb():
    # Carica i dati
    df = pd.read_csv("working_dataset2.csv")

    scaler = StandardScaler()

    # Prepara i dati per "PART 1-2"
    X1 = df[["NUM_CRIMES_TYPE", "NUM_CRIMES_CODE", "NUM_CRIMES_TYPE_DESC"]].copy()
    y1 = df["PART 1-2"]
    X1 = pd.DataFrame(scaler.fit_transform(X1), columns=X1.columns)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "PART 1-2"
    model1 = tune_model(X1_train, y1_train, LGBMClassifier(verbose=-1), param_lgbm())
    best_model1, mean_f1_1, mean_accuracy1, mean_precision1, mean_recall1 = k_fold(X1_train.reset_index(drop=True),
      y1_train.reset_index(drop=True), 5, model1)
    print_metrics(y1_test, best_model1.predict(X1_test), "PART 1-2")
    print_feature_importances(best_model1, X1.columns, "PART 1-2")
    plot_learning_curve(best_model1, X1_train, y1_train,"GB_PART1-2")


    # Prepara i dati per "CRM CD"
    X2 = df[["NUM_CRIMES_CODE", "NUM_CRIMES_TYPE_DESC"]].copy()
    X2["PART 1-2"] = best_model1.predict(X1)
    y2 = df["CRM CD"]
    X2 = pd.DataFrame(scaler.fit_transform(X2), columns=X2.columns)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "CRM CD"
    model2 = tune_model(X2_train, y2_train, LGBMClassifier(verbose=-1), param_lgbm())
    best_model2, mean_f1_2, mean_accuracy2, mean_precision2, mean_recall2 = k_fold(X2_train.reset_index(drop=True), y2_train.reset_index(drop=True), 5, model2)
    print_metrics(y2_test, best_model2.predict(X2_test), "CRM CD")
    print_feature_importances(best_model2, X2.columns, "CRM CD")
    plot_learning_curve(best_model2, X2_train, y2_train,"GB_CRM-CD")


    # Prepara i dati per "AREA"
    X3 = df[["NUM_CRIMES_AREA", "NUM_CRIMES_AREA_NAME", "NUM_CRIMES_DISTRICT"]].copy()
    y3 = df["AREA"]
    X3 = pd.DataFrame(scaler.fit_transform(X3), columns=X3.columns)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "AREA"
    model3 = tune_model(X3_train, y3_train, LGBMClassifier(verbose=-1), param_lgbm())
    best_model3, mean_f1_3, mean_accuracy3, mean_precision3, mean_recall3 = k_fold(X3_train.reset_index(drop=True),
      y3_train.reset_index(drop=True), 5, model3)
    print_metrics(y3_test, best_model3.predict(X3_test), "AREA")
    print_feature_importances(best_model3, X3.columns, "AREA")
    plot_learning_curve(best_model3, X3_train, y3_train,"GB_AREA")


    # Prepara i dati per "RPT DIST NO"
    X4 = df[["NUM_CRIMES_DISTRICT", "NUM_CRIMES_DATE"]].copy()
    X4["AREA"] = best_model3.predict(X3)
    y4 = df["RPT DIST NO"]
    X4 = pd.DataFrame(scaler.fit_transform(X4), columns=X4.columns)
    X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "RPT DIST NO"
    model4 = tune_model(X4_train, y4_train, LGBMClassifier(verbose=-1), param_lgbm())
    best_model4, mean_f1_4, mean_accuracy4, mean_precision4, mean_recall4 = k_fold(X4_train.reset_index(drop=True),
                                                                                   y4_train.reset_index(drop=True), 5,
                                                                                   model4)
    print_metrics(y4_test, best_model4.predict(X4_test), "RPT DIST NO")
    print_feature_importances(best_model4, X4.columns, "RPT DIST NO")
    plot_learning_curve(best_model4, X4_train, y4_train,"GB_RPT-DIST-NO")


    # Prepara i dati per "PREMISE_CODE"
    X5 = df[["NUM_CRIMES_PREMISE_DESC", "NUM_CRIMES_PREMISE", "NUM_CRIMES_DISTRICT"]].copy()
    y5 = df["PREMISE_CODE"]
    X5 = pd.DataFrame(scaler.fit_transform(X5), columns=X5.columns)
    X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.1, random_state=33)

    # Addestra il modello per prevedere "PREMISE_CODE"
    model5 = tune_model(X5_train, y5_train, LGBMClassifier(verbose=-1), param_lgbm())
    best_model5, mean_f1_5, mean_accuracy5, mean_precision5, mean_recall5 = k_fold(X5_train.reset_index(drop=True), y5_train.reset_index(drop=True), 5, model5)
    print_metrics(y5_test, best_model5.predict(X5_test), "PREMISE_CODE")
    print_feature_importances(best_model5, X5.columns, "PREMISE_CODE")
    plot_learning_curve(best_model5, X5_train, y5_train,"GB_PREMISE-CODE")

def cross_validation(df, model, target_col, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    best_f1 = 0  # Migliore F1 score
    best_model = None  # Migliore model
    f1 = 0  # F1 score
    accuracy = 0  # Accuracy
    precision = 0  # Precision
    recall = 0  # Recall

    for train_index, test_index in kf.split(df):
        curr_model = BayesianNetwork(model.edges())
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]

        # Addestra il modello sul set di addestramento
        curr_model.fit(train_df)

        # Utilizza il modello salvato per fare previsioni
        test_df_reduced = test_df[list(curr_model.nodes())]
        y_test = test_df_reduced[target_col]

        y_pred = curr_model.predict(test_df_reduced.drop(columns=[target_col]))

        # Calcola l'accuratezza del modello
        curr_accuracy = accuracy_score(y_test, y_pred)
        accuracy += curr_accuracy

        # Calcola precision, recall e F1 score
        curr_precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        precision += curr_precision
        curr_recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        recall += curr_recall
        curr_f1 = f1_score(y_test, y_pred, average='weighted')
        f1 += curr_f1

        # Check if current model is better than the best model so far
        if curr_f1 > best_f1:
            best_model = curr_model
            best_f1 = curr_f1

    mean_f1 = f1 / n_splits
    mean_accuracy = accuracy / n_splits
    mean_precision = precision / n_splits
    mean_recall = recall / n_splits

    return best_model, mean_f1, mean_accuracy, mean_precision, mean_recall

def evaluate_model(df, model, target_col, test_size=0.1):
    # Dividi i dati in set di addestramento e test
    df_train, df_test = train_test_split(df, test_size=test_size)

    df_train = df_train[list(model.nodes())]
    df_test = df_test[list(model.nodes())]

    # Addestra il modello sul set di addestramento
    model.fit(df_train)

    # Valuta il modello sul set di addestramento
    y_train = df_train[target_col]
    y_train_pred = model.predict(df_train.drop(columns=[target_col]))

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=1)
    train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=1)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')

    # Valuta il modello sul set di test
    y_test = df_test[target_col]
    y_test_pred = model.predict(df_test.drop(columns=[target_col]))
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=1)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=1)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    return train_accuracy, train_precision, train_recall, train_f1, test_accuracy, test_precision, test_recall, test_f1

def reti_bayesiane():
    # Caricamento del dataset
    df = pd.read_csv('working_dataset2.csv')
    df = df[['NUM_CRIMES_DATE', 'NUM_CRIMES_AREA', 'NUM_CRIMES_AREA_NAME', 'NUM_CRIMES_TYPE',
             'NUM_CRIMES_TYPE_DESC', 'NUM_CRIMES_DISTRICT', 'NUM_CRIMES_CODE',
             'NUM_CRIMES_PREMISE', 'NUM_CRIMES_PREMISE_DESC', 'RPT DIST NO',
             'PREMISE_CODE', 'PART 1-2', 'CRM CD', 'AREA']]

    discretizzare = ['NUM_CRIMES_DATE', 'NUM_CRIMES_AREA', 'NUM_CRIMES_AREA_NAME', 'NUM_CRIMES_TYPE',
                     'NUM_CRIMES_TYPE_DESC', 'NUM_CRIMES_DISTRICT', 'NUM_CRIMES_CODE',
                     'NUM_CRIMES_PREMISE', 'NUM_CRIMES_PREMISE_DESC', 'RPT DIST NO',
                     'PREMISE_CODE', 'PART 1-2', 'CRM CD', 'AREA']

    # Discretizzazione delle colonne
    for col in discretizzare:
        df[col] = pd.cut(df[col], bins=10, labels=False)

    df_sample = df

    # Addestramento dei modelli per ciascuna colonna di output
    for col in ['AREA', 'RPT DIST NO', 'PREMISE_CODE', 'PART 1-2', 'CRM CD']:
        df_sample_copy = df_sample.copy()
        if col == 'CRM CD':
            model = BayesianNetwork([('NUM_CRIMES_TYPE_DESC', col),
                                     ('NUM_CRIMES_CODE', col),
                                     ('PART 1-2', col)])
            df_sample_copy = df_sample_copy.drop(columns=['AREA','RPT DIST NO', 'PREMISE_CODE'])
        elif col == 'RPT DIST NO':
            model = BayesianNetwork([('NUM_CRIMES_DISTRICT', col),
                                     ('NUM_CRIMES_DATE', col)])
            df_sample_copy = df_sample_copy.drop(columns=['PART 1-2', 'CRM CD', 'PREMISE_CODE'])
        elif col == 'PREMISE_CODE':
            model = BayesianNetwork([('NUM_CRIMES_PREMISE', col),
                                     ('NUM_CRIMES_PREMISE_DESC', col),
                                     ('NUM_CRIMES_DISTRICT', col)])
            df_sample_copy = df_sample_copy.drop(columns=['PART 1-2', 'CRM CD', 'RPT DIST NO'])
        elif col == 'PART 1-2':
            model = BayesianNetwork([('NUM_CRIMES_TYPE', col),
                                     ('NUM_CRIMES_TYPE_DESC', col),
                                     ('NUM_CRIMES_CODE', col)])
            df_sample_copy = df_sample_copy.drop(columns=['CRM CD', 'AREA', 'RPT DIST NO', 'PREMISE_CODE'])
        else:
            model = BayesianNetwork([('NUM_CRIMES_AREA', col),
                                     ('NUM_CRIMES_AREA_NAME', col),
                                     ('NUM_CRIMES_DISTRICT', col)])

            df_sample_copy = df_sample_copy.drop(columns=['CRM CD', 'PART 1-2', 'RPT DIST NO', 'PREMISE_CODE'])

        best_model,accuracy, precision, recall, f1 = cross_validation(df_sample_copy, model, col)
        print(f"  Metriche di convalida incrociata per {col}:")
        print(f"  Accuracy: {accuracy}")
        print(f"  Precision: {precision}")
        print(f"  Recall: {recall}")
        print(f"  F1 Score: {f1}")


        train_accuracy, train_precision, train_recall, train_f1, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(df_sample_copy, model, col)
        print(f"\nMetriche di addestramento per {col}:")
        print(f"  Accuracy: {train_accuracy}")
        print(f"  Precision: {train_precision}")
        print(f"  Recall: {train_recall}")
        print(f"  F1 Score: {train_f1}")

        print(f"\nMetriche di test per {col}:")
        print(f"  Accuracy: {test_accuracy}")
        print(f"  Precision: {test_precision}")
        print(f"  Recall: {test_recall}")
        print(f"  F1 Score: {test_f1}")

def apprendimento_supevisionato():
    my_rf()
    my_dt()
    my_ada()
    my_gb()
reti_bayesiane()