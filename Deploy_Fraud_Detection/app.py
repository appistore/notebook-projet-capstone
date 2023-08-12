import pandas as pd
import pickle
import sklearn

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    
    file_path = "images/" + imagefile.filename
    print(file_path)
    
    df_X_test = pd.read_csv(file_path)
    print(df_X_test)
    
    # load the model from disk
    #model_rf_clf = pickle.load(open('Models/fraude_detection_model_rf_clf_17-07-2023.sav', 'rb'))
    #model_DNN = pickle.load(open('Models/model_DNN_10_08_2023.sav', 'rb'))
    model_rf_clf = pickle.load(open('Models/fraude_detection_model_rf_clf_10-08-2023.sav', 'rb'))
    
    #result = model_rf_clf.score(X_test, y_test)
    #print(model_rf_clf)
    
    y_reel = df_X_test['isFraud']
    X = df_X_test.drop(['isFraud'], axis=1)
    y_reel = y_reel[0]
    print(y_reel)
    print(X)
    
    y_pred_rf_clf = model_rf_clf.predict(X)
    y_pred_rf_clf
    print(y_pred_rf_clf)
    
    y_proba_predict = model_rf_clf.predict_proba(X)
    print(y_pred_rf_clf)
    print(f"y_proba_predict: {y_proba_predict}")
    
    df_prediction_rfc = pd.DataFrame(y_pred_rf_clf)
    print(df_prediction_rfc)
    df = df_prediction_rfc.merge(df_X_test,how ='left',left_index=True,right_index=True)
    print(df)
    
    isFraud = df[0]
    print(isFraud)
    
    

    if y_pred_rf_clf==[1]:
        return render_template('index.html',y_reel=y_reel, y_pred_rf_clf=y_pred_rf_clf,y_proba_predict=y_proba_predict[1], prediction='This Transaction is Fraud', )
    else:
        return render_template('index.html',y_reel=y_reel, y_pred_rf_clf=y_pred_rf_clf,y_proba_predict=y_proba_predict[0], prediction='This Transaction is Not Fraud')


if __name__ == '__main__':
    app.run(port=3000, debug=True)