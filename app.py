# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:54:53 2024

@author: narze Nishant
"""





import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
app = Flask(__name__)

model = joblib.load("symp.pkl")
vc=joblib.load("vector.pkl")



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global df
    
    input_features = [x for x in request.form.values()]
    #print(input_features)
    new_review = input_features[0]
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    test=vc.transform(new_corpus).toarray()
    
    d_array=['ACNE', 'ARTHRITIS', 'BRONCHIAL ASTHMA', 'CERVICAL SPONDYLOSIS', 'CHICKEN POX', 'COMMON COLD', 'DENGUE', 'DIMORPHIC HEMORRHOIDS', 'FUNGAL INFECTION', 'HYPERTENSION', 'IMPETIGO', 'JAUNDICE', 'MALARIA', 'MIGRAINE', 'PNEUMONIA', 'PSORIASIS', 'TYPHOID', 'VARICOSE VEINS', 'ALLERGY', 'DIABETES', 'DRUG REACTION', 'GASTROESOPHAGEAL REFLUX DISEASE', 'PEPTIC ULCER DISEASE', 'URINARY TRACT INFECTION']
    ind=model.predict(test)[0]
    output=d_array[ind]
    s_array = [
    'Maintain good skincare habits, use topical treatments, and consider prescription medications if necessary.',
    'Depending on the type, treatment may include medications, physical therapy, and lifestyle changes.',
    'Use inhalers as prescribed, identify and avoid triggers, and follow an asthma action plan.',
    'Physical therapy, pain medications, and lifestyle modifications may be recommended.',
    'Rest, antiviral medications, and relieving symptoms with over-the-counter remedies.',
    'Rest, stay hydrated, and use over-the-counter medications to alleviate symptoms.',
    'Manage symptoms with rest, hydration, and medical monitoring. Severe cases may require hospitalization.',
    'Dietary changes, topical treatments, and in some cases, surgical intervention.',
    'Antifungal medications, proper hygiene, and avoiding damp environments.',
    'Lifestyle changes, medications, and regular monitoring of blood pressure.',
    'Antibiotics, good hygiene, and keeping affected areas clean.',
    'Treatment depends on the underlying cause. Rest, hydration, and addressing the cause.',
    'Antimalarial medications, rest, and supportive care.',
    'Identify triggers, use medications as prescribed, and consider lifestyle changes.',
    'Antibiotics, rest, and supportive care.',
    'Topical treatments, medications, and lifestyle adjustments.',
    'Antibiotics, hydration, and rest.',
    'Compression stockings, lifestyle changes, and in some cases, medical procedures.',
    'Avoid allergens, use antihistamines, and consider immunotherapy.',
    'Medications, insulin therapy, dietary changes, and regular monitoring of blood sugar levels.',
    'Discontinue the offending drug, seek medical attention if severe.',
    'Medications, lifestyle changes, and, in severe cases, surgery.',
    'Medications to reduce stomach acid, antibiotics for H. pylori infection, and lifestyle changes.',
    'Antibiotics, increased fluid intake, and good hygiene practices.']
    sol_out=s_array[ind]
    return render_template('result.html', prediction_text='{}'.format(output),sol_text='{}'.format(sol_out))
   
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
    