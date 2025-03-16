from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, RidgeClassifier, LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler  # Optional, for scaling input features
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import os, joblib


global filename
global classifier
global X, y, X_train, X_test, y_train, y_test ,Predictions
global df, df2, sc
global le, labels

labels=['Underperforming Workers','Efficient Workers']

def upload():
    global filename
    global df
    filename = filedialog.askopenfilename(initialdir = "Datasets")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    df= pd.read_csv(filename)
    text.insert(END,'\n\nPredicting Dataset: \n', str(df.head()))
    text.insert(END,df.head())

def preprocess():
    global df
    global X, y, X_train, X_test, y_train, y_test, scaler
    text.delete('1.0', END)

    # Display basic information about the dataset
    #text.insert(END, '\n\nInformation of the dataset: \n', str(df.info()))
    print(df.info())
    text.insert(END, '\n\nDescription of the dataset: \n' + str(df.describe().T))
    text.insert(END, '\n\nChecking null values in the dataset: \n' + str(df.isnull().sum()))
    
    df['wip']=df['wip'].fillna(0)
    
    df['Employes Productivity'].unique()
    
    X = df.drop(['Employes Productivity'], axis = 1)
    y = df['Employes Productivity']
    
    sns.set(style="darkgrid") 
    plt.figure(figsize=(12, 6)) 
    ax = sns.countplot(x=y, data=df)
    plt.title("Count Plot")  
    plt.xlabel("Categories") 
    plt.ylabel("Count") 

    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.show()  
    
    labels = ['Underperforming Workers','Efficient Workers']
    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_poly_scaled, y, test_size=0.2, random_state=44)
    text.insert(END, "\n\nTotal Records used for training: " + str(len(X_train)) + "\n")
    text.insert(END, "\n\nTotal Records used for testing: " + str(len(X_test)) + "\n\n")
    
    plt.figure(figsize=(14, 14))
    sns.set(font_scale=1)
    sns.heatmap(df.corr(), cmap='GnBu_r', annot=True, square=True, linewidths=.5)
    plt.title('Variable Correlation')
    plt.show()


    
mae_list = []
mse_list = []
rmse_list = []
r2_list = []

def PerformanceMetrics(algorithm, predict, testY):
    global X_train, X_test, y_train, y_test 
    # Regression metrics
    mae = mean_absolute_error(testY, predict)
    mse = mean_squared_error(testY, predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(testY, predict)
    
    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)
    
    print(f"{algorithm} Mean Absolute Error (MAE): {mae:.2f}")
    print(f"{algorithm} Mean Squared Error (MSE): {mse:.2f}")
    print(f"{algorithm} Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"{algorithm} R-squared (R²): {r2:.2f}")
    
    text.insert(END, "Performance Metrics of " + str(algorithm) + "\n")
    text.insert(END, "Mean Absolute Error (MAE): " + str(mae) + "\n")
    text.insert(END, "Mean Squared Error (MSE): " + str(mse) + "\n")
    text.insert(END, "Root Mean Squared Error (RMSE): " + str(rmse) + "\n")
    text.insert(END, "R-squared (R²): " + str(r2) + "\n\n")
    # Convert to pandas Series for better compatibility with seaborn
    testY_series = pd.Series(testY.ravel())  # Ensure it's a 1-D array
    predict_series = pd.Series(predict.ravel())  # Ensure it's a 1-D array

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=testY_series, y=predict_series, alpha=0.6)
    plt.plot([min(testY_series), max(testY_series)], [min(testY_series), max(testY_series)], 'r--', lw=2)  # Line of equality
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(algorithm)
    plt.grid(True)
    plt.show()


precision = []
recall = []
fscore = []
accuracy = []

#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, testY,predict):
    global labels
    
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100 
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+' Accuracy    : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' F1-SCORE      : '+str(f))
    text.insert(END, "Performance Metrics of " + str(algorithm) + "\n")
    text.insert(END, "Accuracy: " + str(a) + "\n")
    text.insert(END, "Precision: " + str(p) + "\n")
    text.insert(END, "Recall: " + str(r) + "\n")
    text.insert(END, "F1-SCORE: " + str(f) + "\n\n")
    report=classification_report(predict, testY,target_names=labels)
    print('\n',algorithm+" classification report\n",report)
    text.insert(END, "classification report: \n" + str(report) + "\n\n")
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="Blues" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    
def RFReg():
    global rfr_model, predict, df, m_train, m_test, n_train, n_test
    
    m = df.drop(['actual_productivity'], axis = 1)
    n = df['actual_productivity']
    m_train, m_test, n_train, n_test = train_test_split(m, n, test_size=0.2, random_state=44)

    path = os.path.join('model', 'RFRegressor.pkl')
    if os.path.exists(path):
        rfr_model = joblib.load(path)
        print("Model loaded successfully.")
        # Predict on test data
        predict = rfr_model.predict(m_test)
        PerformanceMetrics("Random Forest Regressor", predict, n_test)
    else:
        rfr_model = RandomForestRegressor()
        rfr_model.fit(m_train, n_train)
        # Save the trained model to a file
        joblib.dump(rfr_model, path)
        print("Model saved successfully.")
        # Predict on test data
        predict = rfr_model.predict(m_test)
        PerformanceMetrics("Random Forest Regressor", predict, n_test)
        
def RidgeModel():
    global mod1, X_train, X_test, y_train, y_test
    global predict

    model_folder = 'model'
    path = os.path.join(model_folder, 'ridge_model.pkl')
    if os.path.exists(path):
        print("Loading existing model...")
        mod1 = joblib.load(path)
        predict=mod1.predict(X_test)
        calculateMetrics("Ridge Classifier",predict,y_test)
    else:
        print("Training a new Ridge Classifier model...")
        mod1 = RidgeClassifier()
        mod1.fit(X_train, y_train)
        joblib.dump(mod1, path)
        print(f"Model saved to {path}")
        y_pred = mod1.predict(X_test)
        calculateMetrics("Ridge Classifier",y_pred,y_test)
'''       
def DNN():
    global #m_train, m_test, n_train, n_test, model, rfr, extractor

    import os
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib

    print('m_train:', m_train.shape)
    print('m_test:', m_test.shape)
    print('n_train:', n_train.shape)
    print('n_test:', n_test.shape)

    model_folder = "model"
    dnn_model_path = os.path.join(model_folder, "dnn_feature_extractor.h5")
    rfr_model_path = os.path.join(model_folder, "rfr_model.pkl")

    if os.path.exists(dnn_model_path) and os.path.exists(rfr_model_path):
        print("Loading saved models...")
        extractor = load_model(dnn_model_path)
        rfr = joblib.load(rfr_model_path)
        m_train_features = extractor.predict(m_train)
        m_test_features = extractor.predict(m_test)
        y_pred = rfr.predict(m_test_features)

        mse = mean_squared_error(n_test, y_pred)
        r2 = r2_score(n_test, y_pred)
        print(f"Random Forest Regressor MSE: {mse:.2f}")
        print(f"Random Forest Regressor R²: {r2:.2f}")

        text.insert(tkinter.END, '\n\n---------DNN Model---------\n\n')   
        text.insert(tkinter.END, f"MSE: {mse:.2f}, R²: {r2:.2f}\n")
        PerformanceMetrics("Random Forest Regressor", y_pred, n_test)
    else:
        print("Training models from scratch...")
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(m_train.shape[1],)))  # Adjust input shape as per your data
        model.add(Dense(32, activation='relu'))  # Feature extraction layer
        model.add(Dense(1, activation='linear'))  # Output layer for regression

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        model.fit(m_train, n_train, epochs=50, verbose=0)

        extractor = Sequential(model.layers[:-1])  # Remove the output layer
        extractor.save(dnn_model_path)
        
        m_train_features = extractor.predict(m_train)
        m_test_features = extractor.predict(m_test)
    
        rfr = RandomForestRegressor()
        rfr.fit(m_train_features, n_train)
         
        joblib.dump(rfr, rfr_model_path)
        
        y_pred = rfr.predict(m_test_features)

        mse = mean_squared_error(n_test, y_pred)
        r2 = r2_score(n_test, y_pred)
        print(f"Random Forest Regressor MSE: {mse:.2f}")
        print(f"Random Forest Regressor R²: {r2:.2f}")

        text.insert(tkinter.END, '\n\n---------DNN Model---------\n\n')   
        text.insert(tkinter.END, f"MSE: {mse:.2f}, R²: {r2:.2f}\n")
        PerformanceMetrics("Random Forest Regressor", y_pred, n_test)
        
'''     
def DNN():
    global X_train, X_test, y_train, y_test, model, rfc, extractor
    
    import os
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense
    from sklearn.metrics import accuracy_score
    import joblib
    
    print('X_train:', X_train.shape)
    print('X_test:', X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:', y_test.shape)

    model_folder = "model"
    dnn_model_path = os.path.join(model_folder, "dnn_feature_extractor.h5")
    rfc_model_path = os.path.join(model_folder, "rfc_model.pkl")

    if os.path.exists(dnn_model_path) and os.path.exists(rfc_model_path):
        print("Loading saved models...")
        extractor = load_model(dnn_model_path)
        rfc = joblib.load(rfc_model_path)
        X_train_features = extractor.predict(X_train)
        X_test_features = extractor.predict(X_test)
        y_pred = rfc.predict(X_test_features)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Random Forest Classifier Accuracy: {accuracy * 100:.2f}%")

        text.insert(tkinter.END, '\n\n---------DNN Model---------\n\n')   
        calculateMetrics("DNN Model", y_test, y_pred)
    else:
        print("Training models from scratch...")
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))  # Adjust input shape as per your data
        model.add(Dense(32, activation='relu'))  # Feature extraction layer
        model.add(Dense(2, activation='sigmoid'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=50, verbose=0)

        extractor = Sequential(model.layers[:-1])  # Remove the output layer
        extractor.save(dnn_model_path)
        
        X_train_features = extractor.predict(X_train)
        X_test_features = extractor.predict(X_test)
    
        rfc = RandomForestClassifier()
        rfc.fit(X_train_features, y_train)
         
        joblib.dump(rfc, rfc_model_path)
        
        y_pred = rfc.predict(X_test_features)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Random Forest Classifier Accuracy: {accuracy * 100:.2f}%")

        text.insert(tkinter.END, '\n\n---------DNN Model---------\n\n')   
        calculateMetrics("DNN Model", y_test, y_pred)

               
def predict():
    global scaler, rfc, model, labels, extractor, mod1

    file = filedialog.askopenfilename(initialdir="Datasets")
    test = pd.read_csv(file)
    
    # Display loaded test data
    text.delete('1.0', END)
    text.insert(END, f'{file} Loaded\n')
    text.insert(END, "\n\nLoaded test data: \n" + str(test) + "\n")
        
    if 'actual_productivity' in test.columns:
        test2 = test.drop(['actual_productivity'], axis=1)
        
    test_values = test.values
    
    test_scaled = scaler.transform(test_values)
    mod_predictions = mod1.predict(test_values)
    DNN_predictions = extractor.predict(test_values)
    print('DNN_predictions:-', DNN_predictions)
    
    predicted_classes = DNN_predictions.argmax(axis=1)
    print('predicted_classes:-', predicted_classes)
    
    predicted_classes = rfc.predict(DNN_predictions)
    
    predicted_labels = [labels[p] for p in mod_predictions]
    
    test['Predicted productivity'] = predicted_labels
    
    test2['Predicted productivity'] = predicted_labels
    
    test3 = test2
    
    le = LabelEncoder()
    test3['Predicted productivity'] = le.fit_transform(test3['Predicted productivity'])
    
    predict = rfr_model.predict(test3)
    
    test3['severity productivity'] = predict
    
    text.insert(END, "\n\nModel Predicted value in test data: \n" + str(test) + "\n")
    text.insert(END, "\n\nModel Predicted value in test data: \n" + str(test3) + "\n")

def graph():
    columns = ["Algorithm Name", "Accuracy", "Precision", "Recall", "f1-score"]
    algorithm_names = ["Ridge Classifier", "DNN Classifier"]
    
    # Combine metrics into a DataFrame
    values = []
    for i in range(len(algorithm_names)):
        values.append([algorithm_names[i], accuracy[i], precision[i], recall[i], fscore[i]])
    
    temp = pd.DataFrame(values, columns=columns)
    text.delete('1.0', END)
    # Insert the DataFrame in the text console
    text.insert(END, "All Model Performance metrics:\n")
    text.insert(END, str(temp) + "\n")
    
    # Plotting the performance metrics
    metrics = ["Accuracy", "Precision", "Recall", "f1-score"]
    index = np.arange(len(algorithm_names))  # Positions of the bars

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2  # Width of the bars
    opacity = 0.8

    # Plotting each metric with an offset
    plt.bar(index, accuracy, bar_width, alpha=opacity, color='b', label='Accuracy')
    plt.bar(index + bar_width, precision, bar_width, alpha=opacity, color='g', label='Precision')
    plt.bar(index + 2 * bar_width, recall, bar_width, alpha=opacity, color='r', label='Recall')
    plt.bar(index + 3 * bar_width, fscore, bar_width, alpha=opacity, color='y', label='f1-score')

    # Labeling the chart
    plt.xlabel('Algorithm')
    plt.ylabel('Scores')
    plt.title('Performance Comparison of All Models')
    plt.xticks(index + bar_width, algorithm_names)  # Setting the labels for x-axis (algorithms)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()



def close():
  main.destroy()

from tkinter import *
main = Tk()
main.title("Predicting Remote Work Productivity")
main.geometry("1000x850")
main.config(bg='light steel blue')

font = ('Times New Roman', 15, 'bold')
title = Label(main, text='Deep Learning with Work Patterns-based Remote Work Productivity Analyzer', 
              justify=CENTER, bg='deepskyblue', fg='white')  # Updated colors for title
title.config(font=font, height=2, width=100)
title.pack(pady=10)

button_frame = Frame(main, bg='light steel blue')
button_frame.pack(pady=20)

font1 = ('Times New Roman', 12, 'bold')  # Changed button font to Times New Roman

# Helper function to create buttons in a grid layout
def create_button(text, command, row, col):
    Button(button_frame, text=text, command=command, bg='lightblue', fg='black', 
           activebackground='lavender', font=font1, width=25).grid(row=row, column=col, padx=15, pady=15)

# First row of buttons
create_button("Upload Productivity Dataset", upload, 0, 0)
create_button("Data Analysis and Preprocessing", preprocess, 0, 1)
create_button("Random Forest Regressor", RFReg, 0, 2)

# Second row of buttons
create_button("Ridge Classifier", RidgeModel, 1, 0)
create_button("DNN Model", DNN, 1, 1)
create_button("Performance Metrics Graph", graph, 1, 2)

# Third row of buttons
create_button("Prediction on Test Data", predict, 2, 0)
create_button("Close Application", close, 2, 1)
# Optionally add more buttons here if needed

# Text Box with Scrollbar (placed at the bottom for displaying results/logs)
text_frame = Frame(main, bg='lavender')
text_frame.pack(pady=20)  # Padding for spacing

# Updated Text Box and Scrollbar appearance
text = Text(text_frame, height=25, width=125, wrap=WORD, bg='white', fg='black', font=('Times New Roman', 12))  # Changed font to Times New Roman
scroll = Scrollbar(text_frame, command=text.yview)
text.configure(yscrollcommand=scroll.set)

text.pack(side=LEFT, fill=BOTH, expand=True)
scroll.pack(side=RIGHT, fill=Y)

# Run the main loop
main.mainloop()