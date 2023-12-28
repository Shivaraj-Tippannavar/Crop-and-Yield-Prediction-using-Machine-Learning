
from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np
#import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('crop_recommendation.csv')

features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
labels = df['label']
acc = []
model = []




app = Flask(__name__)

@app.route("/")
def home():
    from sklearn.model_selection import train_test_split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

    from sklearn.tree import DecisionTreeClassifier
    DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)
    DecisionTree.fit(Xtrain,Ytrain)

    predicted_values = DecisionTree.predict(Xtest)
    x = metrics.accuracy_score(Ytest, predicted_values)
    acc.append(x)
    model.append('Decision Tree')

    DT_pkl_filename = 'DecisionTree.pkl'
    
    DT_Model_pkl = open(DT_pkl_filename, 'wb')
    pickle.dump(DecisionTree, DT_Model_pkl)
    return render_template("home.php")

@app.route("/predict",methods=["GET","POST"])
def predict():
    N = request.form['N']
    if N.isdigit()==False:
        N=0
        
    N=float(N)

    P = request.form['P']
    if P.isdigit()==False:
        P=0
    P = float( P)

    K = request.form['K']
    if K.isdigit()==False:
        K=0
    K = float(K)

    temperature = request.form['temperature']
    if temperature.isdigit()==False:
        temperature=0
    temperature= float(temperature)

    humidity= request.form['humidity']
    if  humidity.isdigit()==False:
         humidity=0
    humidity = float(humidity)

    ph = request.form['ph']
    if ph.isdigit()==False:
        ph=0
    ph=float(ph)

    rainfall = request.form['rainfall']
    if rainfall.isdigit()==False:
        rainfall=0
    rainfall = float( rainfall)

    form_array = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    model = pickle.load(open("DecisionTree.pkl","rb"))
    prediction = model.predict(form_array)[0]
    result = prediction
    pic="lentil.jpg"
    if result =="rice":
        result="rice (ಅಕ್ಕಿ)"
        pic='rice.jpg'
    if result =="maize":
        result="maize (ಮೆಕ್ಕೆ ಜೋಳ)"
        pic='maize.jpg'
    if result =="chickpea":
        result="chickpea (ಕಡಲೆ)"
        pic='chickpea.jpg'
    if result =="kidneybeans":
        result="kidneybeans (ಮೂತ್ರಪಿಂಡದ ಹುರುಳಿ)"
        pic='kidneybeans.jpg'
    if result =="pigeonpeas":
        result="pigeonpeas (ತೊಗರಿ ಕಾಳು)"
        pic='pigeonpeas.jpg'
    if result =="mothbeans":
        result="mothbeans (ಮಡಕಿ ಕಾಳು)"
        pic='mothbeans.jpg'
    if result =="mungbean":
        result="mungbean (ಹೆಸರು ಕಾಳು)"
        pic='mungbean.jpg'
    if result =="blackgram":
        result="blackgram (ಉದ್ದಿನ ಬೇಳೆ)"
        pic='blackgram.jpg'
    if result =="lentil":
        result="lentil (ಬೇಳೆಕಾಳುಗಳು)"
        pic='lentil.jpg'
    if result =="pomegranate":
        result="pomegranate (ದಾಳಿಂಬೆ)"
        pic='pomegranate.jpg'
    if result =="banana":
        result="banana (ಬಾಳೆಹಣ್ಣು)"
        pic='banana.jpg'
    if result =="mango":
        result="mango (ಮಾವಿನ ಹಣ್ಣು)"
        pic='mango.jpg'
    if result =="grapes":
        result="grapes (ದ್ರಾಕ್ಷಿ)"
        pic='grapes.jpg'
    if result =="watermelon":
        result="watermelon (ಕಲ್ಲಂಗಡಿ)"
        pic='watermelon.jpg'
    if result =="muskmelon":
        result="muskmelon (ಕರಬೂಜ ಹಣ್ಣು)"
        pic='muskmelon.jpg'
    if result =="apple":
        result="apple (ಸೇಬು ಹಣ್ಣು)"
        pic='apple.jpg'
    if result =="orange":
        result="orange (ಕಿತ್ತಳೆ ಹಣ್ಣು)"
        pic='orange.jpg'
    if result =="papaya":
        result="papaya (ಪಪ್ಪಾಯಿ ಹಣ್ಣು)"
        pic='papaya.jpg'
    if result =="coconut":
        result="coconut (ತೆಂಗಿನಕಾಯಿ)"
        pic='coconut.jpg'
    if result =="cotton":
        result="cotton (ಹತ್ತಿ)"
        pic='cotton.jpg'
    if result =="jute":
        result="jute (ಸೆಣಬು)"
        pic='jute.jpg'
    if result =="coffee":
        result="coffee (ಕಾಫಿ)"
        pic='coffee.jpg'
    
    return render_template("result.html",result = result,pic=pic)

if __name__ == "__main__":
    app.run(debug=True)
