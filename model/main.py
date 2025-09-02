import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def create_model(data, model_type='logistic'):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'logistic':
        model = LogisticRegression()
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
    elif model_type == 'svm':
        model = SVC()
    elif model_type == 'knn':
        model = KNeighborsClassifier()
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier()
    else:
        raise ValueError("Invalid model type specified")
    

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy of our model: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return model, scaler

def get_clean_data():
    data = pd.read_csv('app/data.csv')
    data = data.drop(['Unnamed: 32','id'], axis=1)

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def main():
    data = get_clean_data()
    
    # Create Logistic Regression Model
    logistic_model, logistic_scaler = create_model(data, model_type='logistic')
    with open('model/logistic_model.pkl', 'wb') as f:
        joblib.dump(logistic_model, f)
    with open('model/logistic_scaler.pkl', 'wb') as f:
        joblib.dump(logistic_scaler, f)
    # Create Random Forest Model
    random_forest_model, random_forest_scaler = create_model(data, model_type='random_forest')
    with open('model/random_forest_model.pkl', 'wb') as f:
        joblib.dump(random_forest_model, f)
    with open('model/random_forest_scaler.pkl', 'wb') as f:
        joblib.dump(random_forest_scaler, f)

    # Create Support Vector Machine Model
    svm_model, svm_scaler = create_model(data, model_type='svm')
    with open('model/svm_model.pkl', 'wb') as f:
        joblib.dump(svm_model, f)
    with open('model/svm_scaler.pkl', 'wb') as f:
        joblib.dump(svm_scaler, f)

    # Create K-Nearest Neighbors Model
    knn_model, knn_scaler = create_model(data, model_type='knn')
    with open('model/knn_model.pkl', 'wb') as f:
        joblib.dump(knn_model, f)
    with open('model/knn_scaler.pkl', 'wb') as f:
        joblib.dump(knn_scaler, f)

    # Create Gradient Boosting Classifier Model
    gradient_boosting_model, gradient_boosting_scaler = create_model(data, model_type='gradient_boosting')
    with open('model/gradient_boosting_model.pkl', 'wb') as f:
        joblib.dump(gradient_boosting_model, f)
    with open('model/gradient_boosting_scaler.pkl', 'wb') as f:
        joblib.dump(gradient_boosting_scaler, f)
if __name__ == '__main__':
    main()
