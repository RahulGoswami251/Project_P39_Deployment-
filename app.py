import pandas as pd
from flask import Flask, request,jsonify,render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer


app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
cv=pickle.load(open('trnsform.pkl','rb'))

#cols_when_model_builds = model.get_booster().feature_names

@app.route('/')
def index():
    return render_template('SentimentAnalysis.html',Predicted_val=0)


@app.route('/',methods=['POST'])
def Get_All_Details():    
    if request.method == 'POST':
          
          JwMarriot_reviews = request.form.getlist("review")
          print(JwMarriot_reviews)
          vect=cv.transform(JwMarriot_reviews).toarray()
          cv.transform(['worst service ever star hotel dont pick phone clean roomthey kept call waiting minutes still didnt clean room came back hours worst hotel ever']).toarray()
          print(vect)
          
          pred_df = model.predict(vect)
          
          if pred_df[0] == 0:
             output_lbl='Neutral'
          elif pred_df[0] == 1:
             output_lbl='Positive'
          elif pred_df[0] == 2:
             output_lbl='Negative'
          
          return render_template('Prediction.html',Predicted_val=output_lbl)


       
    
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)