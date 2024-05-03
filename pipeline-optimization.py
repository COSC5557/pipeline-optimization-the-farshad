# %%
from bayes_opt import BayesianOptimization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
import pandas as pd
# %%

# %%
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# %%
data.head()

# %%
data.describe()

# %%
data.info()

# %%
pd.DataFrame(data.isnull().sum())

# %%
plt.figure(figsize=(10, 6), dpi=300)
plt.title('Quality Distribution')
sns.barplot(data['quality'].value_counts())
plt.show()

# %%
plt.figure(figsize=(20, 6), dpi=300)
# plot data distribution for each feature
for i, col in enumerate(data.columns[:-1]):
    plt.subplot(3, 4, i+1)
    sns.histplot(data[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12, 6), dpi=300)
plt.title('Correlation Heatmap of Wine Quality')
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# %%
X = data.drop('quality', axis=1)
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# %%
n_iter = 100
init_points = 5

# %%
models = {
    'Support Vector Machine': SVC(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
}

# %%
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'{name} Accuracy: {accuracy_score(y_test, y_pred)}')
    results[name] = accuracy_score(y_test, y_pred)

plt.figure(figsize=(10, 6), dpi=300)
plt.title('Model Accuracy Before Hyperparameter Tuning')
sns.barplot(x=list(results.keys()), y=list(results.values()), color='skyblue')
plt.ylabel('Accuracy')
plt.show()

# %%


def classifier_pipeline_dv():
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer()),
                ('feature_selection', SelectKBest(f_classif)),
                ('pca', PCA())
            ]), X.columns)
        ])),
        ('classifier', SVC(random_state=42))
    ])
    return pipeline


def eval_svm_pipeline_dv():
    pipeline = classifier_pipeline_dv()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


svm_accuracy_pipeline_dv = eval_svm_pipeline_dv()

# %%
svm_bounds_pipeline = {
    'k': (3, 11),
}


def classifier_pipeline_pl(k):
    k = int(round(k))
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer()),
                ('feature_selection', SelectKBest(f_classif, k=k)),
                ('pca', PCA())
            ]), X.columns)
        ])),
        ('classifier', SVC(random_state=42))
    ])
    return pipeline


def svm_eval_pipeline(k):
    pipeline = classifier_pipeline_pl(k)
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    return scores.mean()


svm_optimizer_pipeline = BayesianOptimization(
    f=svm_eval_pipeline, pbounds=svm_bounds_pipeline, random_state=42)
svm_optimizer_pipeline.maximize(n_iter=n_iter, init_points=init_points)

# %%
target_values = [res['target'] for res in svm_optimizer_pipeline.res]
iterations = list(range(1, len(target_values) + 1))

cumulative_max = np.maximum.accumulate(target_values)

plt.figure(figsize=(10, 6))
plt.plot(iterations, target_values, marker='o',
         color='lightblue', label='Objective Function Value')
plt.plot(iterations, cumulative_max, marker='', color='red',
         linestyle='--', label='Best Value Found')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Bayesian Optimization Process for SVM')
plt.legend()
plt.grid(False)
plt.show()


# %%
def eval_svm_pipeline(k):
    pipeline = classifier_pipeline_pl(k)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


svm_accuracy_pipeline = eval_svm_pipeline(
    **svm_optimizer_pipeline.max['params'])

# %%
svm_bounds = {
    'C': (1, 100),
    'k': (9, 11),
}

# %%


def classifier_pipeline(C, k):
    k = int(round(k))
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer()),
                ('feature_selection', SelectKBest(f_classif, k=k)),
                ('pca', PCA())
            ]), X.columns)
        ])),
        ('classifier', SVC(C=C, gamma='scale', random_state=42))
    ])
    return pipeline

# %%


def svm_eval(C, k=10):
    pipeline = classifier_pipeline(C, k)
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    return scores.mean()


# %%
svm_optimizer = BayesianOptimization(
    f=svm_eval, pbounds=svm_bounds, random_state=42)
svm_optimizer.maximize(n_iter=n_iter, init_points=init_points)

# %%
target_values = [res['target'] for res in svm_optimizer.res]
iterations = list(range(1, len(target_values) + 1))

cumulative_max = np.maximum.accumulate(target_values)

plt.figure(figsize=(10, 6))
plt.plot(iterations, target_values, marker='o',
         color='lightblue', label='Objective Function Value')
plt.plot(iterations, cumulative_max, marker='', color='red',
         linestyle='--', label='Best Value Found')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Bayesian Optimization Process for SVM')
plt.legend()
plt.grid(False)
plt.show()


# %%
best_params = svm_optimizer.max['params']
accuracy = svm_optimizer.max['target']

print(f"Best hyperparameters for SVM:\n"
      f"C: {best_params['C']}\n"
      f"k: {int(round(best_params['k']))}\n"
      f"Accuracy: {accuracy}")


# %%
def eval_svm_pipeline(C, k):
    pipeline = classifier_pipeline(C, k)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


svm_accuracy = eval_svm_pipeline(**svm_optimizer.max['params'])

# %%
print(f'SVM Accuracy Before Hyperparameter Tuning: {
      results["Support Vector Machine"]}')
print(f'SVM Accuracy After Pipeline With Default Value: {
      svm_accuracy_pipeline_dv}')
print(f'SVM Accuracy After Pipeline Parameter Optimization on Validation dataset: {
      svm_optimizer.max["target"]}')
print(f'SVM Accuracy After Pipeline Parameter Optimization: {
      svm_accuracy_pipeline}')
print(f'SVM Accuracy After Hyperparameter Tuning and Pipeline parameter optimization on Validation dataset: {
      svm_optimizer_pipeline.max["target"]}')
print(f'SVM Accuracy After Hyperparameter Tuning and Pipeline parameter optimization: {
      svm_accuracy}')

# %%
# Plot the SVM model accuracy before and after hyperparameter tuning
hfont = {'fontname': 'ubuntu'}

plt.figure(figsize=(10, 6), dpi=300)
plt.title('SVM Accuracy Before and After Hyperparameter Tuning', **hfont)
# font select

plt.rcParams.update({'font.size': 7})
sns.barplot(x=['Without Pipeline', 'Pipeline with Default Value', 'Pipeline Optimization', 'Pipeline Optimization and HPO'], y=[
            results['Support Vector Machine'], svm_accuracy_pipeline_dv, svm_accuracy_pipeline, svm_accuracy], color='skyblue')
plt.ylabel('Accuracy')
plt.show()


# %%
def gb_pipeline_dv():
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer()),
                ('feature_selection', SelectKBest(f_classif)),
                ('pca', PCA())
            ]), X.columns)
        ])),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    return pipeline


def eval_gb_pipeline_dv():
    pipeline = gb_pipeline_dv()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


gb_accuracy_dv = eval_gb_pipeline_dv()

# %%
gb_bounds_pl = {
    'n_components': (0.7, 0.99),
    'k': (3, 11),
}


def gb_pipeline_pl(n_components, k):
    k = int(round(k))
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer()),
                ('feature_selection', SelectKBest(f_classif, k=k)),
                ('pca', PCA(n_components=n_components))
            ]), X.columns)
        ])),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    return pipeline


def gb_eval_pl(n_components, k):
    pipeline = gb_pipeline_pl(n_components, k)
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=5, scoring='accuracy')
    return scores.mean()


gb_optimizer_pl = BayesianOptimization(
    f=gb_eval_pl, pbounds=gb_bounds_pl, random_state=42)
gb_optimizer_pl.maximize(n_iter=n_iter, init_points=init_points)
print("Best hyperparameters for Gradient Boosting:",
      gb_optimizer_pl.max['params'])

# %%


def eval_gb_pipeline(n_components, k):
    pipeline = gb_pipeline_pl(n_components, k)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


gb_accuracy_pl = eval_gb_pipeline(**gb_optimizer_pl.max['params'])

# %%
gb_bounds = {
    'n_estimators': (100, 400),
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3),
    'n_components': (0.7, 0.99),
    'k': (3, 11),
}

# %%


def gb_pipeline(n_estimators, max_depth, learning_rate, n_components, k):
    k = int(round(k))
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer()),
                ('feature_selection', SelectKBest(f_classif, k=k)),
                ('pca', PCA(n_components=n_components))
            ]), X.columns)
        ])),
        ('classifier', GradientBoostingClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            random_state=42
        ))
    ])
    return pipeline

# %%


def gb_eval(n_estimators, max_depth, learning_rate, n_components, k):
    pipeline = gb_pipeline(n_estimators, max_depth,
                           learning_rate, n_components, k)
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=5, scoring='accuracy')
    return scores.mean()


# %%
gb_optimizer = BayesianOptimization(
    f=gb_eval, pbounds=gb_bounds, random_state=42)
gb_optimizer.maximize(n_iter=n_iter, init_points=init_points)
print("Best hyperparameters for Gradient Boosting:",
      gb_optimizer.max['params'])

# %%
target_values = [res['target'] for res in gb_optimizer.res]
iterations = list(range(1, len(target_values) + 1))

cumulative_max = np.maximum.accumulate(target_values)

plt.figure(figsize=(10, 6))
plt.plot(iterations, target_values, marker='o',
         color='lightblue', label='Objective Function Value')
plt.plot(iterations, cumulative_max, marker='', color='red',
         linestyle='--', label='Best Value Found')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Bayesian Optimization Process for Gradient Boosting:')
plt.legend()
plt.grid(False)
plt.show()

# %%
best_params = gb_optimizer.max['params']
accuracy = gb_optimizer.max['target']

print(f"Best hyperparameters for SVM:\n"
      f'n_estimators: {int(round(best_params["n_estimators"]))}\n'
      f'max_depth: {int(round(best_params["max_depth"]))}\n'
      f'learning_rate: {best_params["learning_rate"]}\n'
      f"n_components: {best_params['n_components']}\n"
      f"k: {int(round(best_params['k']))}\n"
      f"Accuracy: {accuracy}")


# %%
def eval_gb_pipeline(n_estimators, max_depth, learning_rate, n_components, k):
    pipeline = gb_pipeline(n_estimators, max_depth,
                           learning_rate, n_components, k)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


gb_accuracy = eval_gb_pipeline(**gb_optimizer.max['params'])

# %%
print(f'Gradient Boosting Accuracy Before Hyperparameter Tuning: {
      results["Gradient Boosting"]}')
print(f'Gradient Boosting Accuracy After Pipeline With Default Value: {
      gb_accuracy_dv}')
print(f'Gradient Boosting Accuracy After Pipeline Parameter Optimization on Validation dataset: {
      gb_optimizer_pl.max["target"]}')
print(f'Gradient Boosting Accuracy After Pipeline Parameter Optimization: {
      gb_accuracy_pl}')
print(f'Gradient Boosting Accuracy After Hyperparameter Tuning and Pipeline parameter optimization on Validation dataset: {
      gb_optimizer.max["target"]}')
print(f'Gradient Accuracy After Hyperparameter Tuning and Pipeline parameter optimization: {
      gb_accuracy}')

# %%
hfont = {'fontname': 'ubuntu'}
plt.figure(figsize=(10, 6), dpi=300)
plt.rcParams.update({'font.size': 7})
plt.title(
    'Gradient Boosting Accuracy Before and After Hyperparameter Tuning',  **hfont)
sns.barplot(x=['Without Pipeline', 'Pipeline with Default Value', 'Pipeline Optimization', 'Pipeline Optimization and HPO'], y=[
            results['Gradient Boosting'], gb_accuracy_dv, gb_accuracy_pl, gb_accuracy], color='skyblue')
plt.ylabel('Accuracy')
plt.show()


# %%
def rf_pipeline_dv():
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer()),
                ('feature_selection', SelectKBest(f_classif)),
                ('pca', PCA())
            ]), X.columns)
        ])),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    return pipeline


def eval_rf_pipeline_dv():
    pipeline = rf_pipeline_dv()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


rf_accuracy_dv = eval_rf_pipeline_dv()

# %%
rf_bound_pl = {
    'n_components': (0.7, 0.99),
    'k': (3, 11),
}


def rf_pipeline_pl(n_components, k):
    k = int(round(k))
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer()),
                ('feature_selection', SelectKBest(f_classif, k=k)),
                ('pca', PCA(n_components=n_components))
            ]), X.columns)
        ])),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    return pipeline


def rf_eval_pl(n_components, k):
    pipeline = rf_pipeline_pl(n_components, k)
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=5, scoring='accuracy')
    return scores.mean()


rf_optimizer_pl = BayesianOptimization(
    f=rf_eval_pl, pbounds=rf_bound_pl, random_state=42)
rf_optimizer_pl.maximize(n_iter=n_iter, init_points=init_points)


def eval_rf_pipeline_pl(n_components, k):
    pipeline = rf_pipeline_pl(n_components, k)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


rf_accuracy_pl = eval_rf_pipeline_pl(**rf_optimizer_pl.max['params'])

# %%
rf_bounds = {
    'n_estimators': (100, 400),
    'max_depth': (3, 10),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 4),
    'n_components': (0.7, 0.99),
    'k': (3, 11),
}


def rf_pipeline(n_estimators, max_depth, min_samples_split, min_samples_leaf, n_components, k):
    k = int(round(k))
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer()),
                ('feature_selection', SelectKBest(f_classif, k=k)),
                ('pca', PCA(n_components=n_components))
            ]), X.columns)
        ])),
        ('classifier', RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            random_state=42
        ))
    ])
    return pipeline


def rf_eval(n_estimators, max_depth, min_samples_split, min_samples_leaf, n_components, k):
    pipeline = rf_pipeline(n_estimators, max_depth,
                           min_samples_split, min_samples_leaf, n_components, k)
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=5, scoring='accuracy')
    return scores.mean()


rf_optimizer = BayesianOptimization(
    f=rf_eval, pbounds=rf_bounds, random_state=42)
rf_optimizer.maximize(n_iter=n_iter, init_points=init_points)
print("Best hyperparameters for Random Forest:", rf_optimizer.max['params'])

target_values = [res['target'] for res in rf_optimizer.res]
iterations = list(range(1, len(target_values) + 1))

cumulative_max = np.maximum.accumulate(target_values)

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(iterations, target_values, marker='o',
         color='lightblue', label='Objective Function Value')
plt.plot(iterations, cumulative_max, marker='', color='red',
         linestyle='--', label='Best Value Found')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Bayesian Optimization Process for Random Forest')
plt.legend()
plt.grid(False)
plt.show()

# %%
best_params = rf_optimizer.max['params']
accuracy = rf_optimizer.max['target']

print(f"Best hyperparameters for SVM:\n"
      f'n_estimators: {int(round(best_params["n_estimators"]))}\n'
      f'max_depth: {int(round(best_params["max_depth"]))}\n'
      f'min_samples_split: {int(round(best_params["min_samples_split"]))}\n'
      f'min_samples_leaf: {int(round(best_params["min_samples_leaf"]))}\n'
      f"k: {int(round(best_params['k']))}\n"
      f"Accuracy: {accuracy}")


# %%
def eval_rf_pipeline(n_estimators, max_depth, min_samples_split, min_samples_leaf, n_components, k):
    pipeline = rf_pipeline(n_estimators, max_depth,
                           min_samples_split, min_samples_leaf, n_components, k)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


rf_accuracy = eval_rf_pipeline(**rf_optimizer.max['params'])

# %%
print(f'Random Forest Accuracy Before Hyperparameter Tuning: {
      results["Random Forest"]}')
print(f'Random Forest Accuracy After Pipeline With Default Value: {
      rf_accuracy_dv}')
print(f'Random Forest Accuracy After Pipeline Parameter Optimization on Validation dataset: {
      rf_optimizer_pl.max["target"]}')
print(f'Random Forest Accuracy After Pipeline Parameter Optimization: {
      rf_accuracy_pl}')
print(f'Random Forest Accuracy After Hyperparameter Tuning and Pipeline parameter optimization on Validation dataset: {
      rf_optimizer.max["target"]}')
print(f'Random Forest Accuracy After Hyperparameter Tuning and Pipeline parameter optimization: {
      rf_accuracy}')

# %%
hfont = {'fontname': 'ubuntu'}
plt.figure(figsize=(10, 6), dpi=300)
plt.rcParams.update({'font.size': 7})
plt.title('Random Forest Accuracy Before and After Hyperparameter Tuning',  **hfont)
sns.barplot(x=['Without Pipeline', 'Pipeline with Default Value', 'Pipeline Optimization', 'Pipeline Optimization and HPO'], y=[
            results['Random Forest'], rf_accuracy_dv, rf_accuracy_pl, rf_accuracy], color='skyblue')
plt.ylabel('Accuracy')
plt.show()


# %%
def dt_pipeline_dv():
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer()),
                ('feature_selection', SelectKBest(f_classif)),
                ('pca', PCA())
            ]), X.columns)
        ])),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    return pipeline


def eval_dt_pipeline_dv():
    pipeline = dt_pipeline_dv()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


dt_accuracy_dv = eval_dt_pipeline_dv()

# %%
dt_bounds_pl = {
    'n_components': (0.7, 0.99),
    'k': (3, 11),
}


def dt_pipeline_pl(n_components, k):
    k = int(round(k))
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer()),
                ('feature_selection', SelectKBest(f_classif, k=k)),
                ('pca', PCA(n_components=n_components))
            ]), X.columns)
        ])),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    return pipeline


def dt_eval_pl(n_components, k):
    pipeline = dt_pipeline_pl(n_components, k)
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=5, scoring='accuracy')
    return scores.mean()


dt_optimizer_pl = BayesianOptimization(
    f=dt_eval_pl, pbounds=dt_bounds_pl, random_state=42)
dt_optimizer_pl.maximize(n_iter=n_iter, init_points=init_points)
print("Best hyperparameters for Decision Tree:", dt_optimizer_pl.max['params'])


def eval_dt_pipeline(n_components, k):
    pipeline = dt_pipeline_pl(n_components, k)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


dt_accuracy_pl = eval_dt_pipeline(**dt_optimizer_pl.max['params'])

# %%
dt_bounds = {
    'max_depth': (3, 20),
    'min_samples_split': (3, 20),
    'min_samples_leaf': (3, 8),
    'n_components': (0.7, 0.99),
    'k': (3, 11),
}


def dt_pipeline(max_depth, min_samples_split, min_samples_leaf, n_components, k):
    k = int(round(k))
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer()),
                ('feature_selection', SelectKBest(f_classif, k=k)),
                ('pca', PCA(n_components=n_components))
            ]), X.columns)
        ])),
        ('classifier', DecisionTreeClassifier(
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            random_state=42
        ))
    ])
    return pipeline


def dt_eval(max_depth, min_samples_split, min_samples_leaf, n_components, k):
    pipeline = dt_pipeline(max_depth, min_samples_split,
                           min_samples_leaf, n_components, k)
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=5, scoring='accuracy')
    return scores.mean()


dt_optimizer = BayesianOptimization(
    f=dt_eval, pbounds=dt_bounds, random_state=42)
dt_optimizer.maximize(n_iter=n_iter, init_points=init_points)
print("Best hyperparameters for Decision Tree:", dt_optimizer.max['params'])

# %%
target_values = [res['target'] for res in dt_optimizer.res]
iterations = list(range(1, len(target_values) + 1))

cumulative_max = np.maximum.accumulate(target_values)

plt.figure(figsize=(10, 6))
plt.plot(iterations, target_values, marker='o',
         color='lightblue', label='Objective Function Value')
plt.plot(iterations, cumulative_max, marker='', color='red',
         linestyle='--', label='Best Value Found')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Bayesian Optimization Process for Decision Tree')
plt.legend()
plt.grid(False)
plt.show()


# %%
best_params = dt_optimizer.max['params']
accuracy = dt_optimizer.max['target']

print(f"Best hyperparameters for Decision Tree:\n"
      f'max_depth: {int(round(best_params["max_depth"]))}\n'
      f'min_samples_split: {int(round(best_params["min_samples_split"]))}\n'
      f'min_samples_leaf: {int(round(best_params["min_samples_leaf"]))}\n'
      f"k: {int(round(best_params['k']))}\n"
      f"Accuracy: {accuracy}")

# %%


def eval_dt_pipeline(max_depth, min_samples_split, min_samples_leaf, n_components, k):
    pipeline = dt_pipeline(max_depth, min_samples_split,
                           min_samples_leaf, n_components, k)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


dt_accuracy = eval_dt_pipeline(**dt_optimizer.max['params'])

# %%
print(f'Decision Tree Accuracy Before Hyperparameter Tuning: {
      results["Decision Tree"]}')
print(f'Decision Tree Accuracy After Pipeline With Default Value: {
      dt_accuracy_dv}')
print(f'Decision Tree Accuracy After Pipeline Parameter Optimization on Validation dataset: {
      dt_optimizer_pl.max["target"]}')
print(f'Decision Tree Accuracy After Pipeline Parameter Optimization: {
      dt_accuracy_pl}')
print(f'Decision Tree Accuracy After Hyperparameter Tuning and Pipeline parameter optimization on Validation dataset: {
      dt_optimizer.max["target"]}')
print(f'Decision Tree Accuracy After Hyperparameter Tuning and Pipeline parameter optimization: {
      dt_accuracy}')

# %%
hfont = {'fontname': 'ubuntu'}
plt.figure(figsize=(10, 6), dpi=300)
plt.rcParams.update({'font.size': 7})
plt.title('Decision Tree Accuracy Before and After Hyperparameter Tuning',  **hfont)
sns.barplot(x=['Without Pipeline', 'Pipeline with Default Value', 'Pipeline Optimization', 'Pipeline Optimization and HPO'],  y=[
            results['Decision Tree'], dt_accuracy_dv, dt_accuracy_pl, dt_accuracy], color='skyblue')
plt.ylabel('Accuracy')
plt.show()

# %%
models_best_results = {
    'SVM': svm_optimizer.max['target'],
    'Gradient Boosting': gb_optimizer.max['target'],
    'Random Forest': rf_optimizer.max['target'],
    'Decision Tree': dt_optimizer.max['target']
}
plt.figure(figsize=(10, 6), dpi=300)
plt.bar(models_best_results.keys(),
        models_best_results.values(), color='lightblue')
plt.title('Model Accuracy Before Hyperparameter Tuning')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()

# %%
final_results = {
    'svm': {
        'Initial Accuracy': results['Support Vector Machine'],
        'Default Pipeline Accuracy': svm_accuracy_pipeline_dv,
        'Optimized Pipeline Accuracy': svm_accuracy_pipeline,
        'Optimized Accuracy': svm_accuracy
    },
    'gb': {
        'Initial Accuracy': results['Gradient Boosting'],
        'Default Pipeline Accuracy': gb_accuracy_dv,
        'Optimized Pipeline Accuracy': gb_accuracy_pl,
        'Optimized Accuracy': gb_accuracy
    },
    'rf': {
        'Initial Accuracy': results['Random Forest'],
        'Default Pipeline Accuracy': rf_accuracy_dv,
        'Optimized Pipeline Accuracy': rf_accuracy_pl,
        'Optimized Accuracy': rf_accuracy
    },
    'dt': {
        'Initial Accuracy': results['Decision Tree'],
        'Default Pipeline Accuracy': dt_accuracy_dv,
        'Optimized Pipeline Accuracy': dt_accuracy_pl,
        'Optimized Accuracy': dt_accuracy
    }
}


# %%
classifiers = list(final_results.keys())
stages = list(next(iter(final_results.values())).keys())
data = {stage: [final_results[classifier][stage]
                for classifier in classifiers] for stage in stages}

n_classifiers = len(classifiers)
n_stages = len(stages)
bar_width = 0.15
index = np.arange(n_classifiers)

fig, ax = plt.subplots()

for i, stage in enumerate(stages):
    ax.bar(index + i*bar_width, data[stage], bar_width, label=stage)

ax.set_xlabel('Classifier')
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracies by Classifier and Stage')
ax.set_xticks(index + bar_width*(n_stages-1)/2)
ax.set_xticklabels(classifiers)
ax.legend()

fig.tight_layout()
plt.show()
