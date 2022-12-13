from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify, make_response, session
from .models import *
from . import db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, current_user

_route_it = Blueprint('_route_it', __name__)

@_route_it.route('/index')
@login_required
def dashboard():
    return '1'
    
    
import numpy as np
import pandas as pd
import pickle
from tkinter import Y
from pandas import DataFrame
from flask import Flask, request, jsonify, render_template
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import spearmanr
# import matplotlib.pyplot as plt
import json
# import plotly
# import plotly.express as px
import random
import math
from os import path
import os

csv_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"model/Institute-of-Computer-Studies-Graduate-Tracer-Study-2021-2022-Responses(ALTERED).csv")

model_IT_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"model/ProjectModel_IT.pkl")
model_CS_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"model/ProjectModel_CS.pkl")
model_ITsuggest_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"model/IT_SUGGESTEDcourse.pkl")
model_CSsuggest_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"model/CS_SUGGESTEDcourse.pkl")

dataset = pd.read_csv(csv_path)

ordinal_encoder = OrdinalEncoder()

IT_FEATURES = [
     'Sex',
     'Shiftee',
     'ComProg_1_1st',
     'ComProg_1_2nd',
     'ComProg_2_1st',
     'ComProg_2_2nd',
     'Intro_to_Computing_1st',
     'Intro_to_Computing_2nd',
     'Info_Management_1st',
     'Info_Management_2nd',
     'Operating_System_1st',
     'Operating_System_2nd',
     'Elective_1_1st',
     'Elective_1_2nd',
     'Elective_2_1st',
     'Elective_2_2nd',
     'Elective_3_1st',
     'Elective_3_2nd',
     'Data_Structures_1st',
     'Data_Structures_2nd',
     'Application_Dev_and_Emerging_Tech_1st',
     'Application_Dev_and_Emerging_Tech_2nd',
     'Human_and_Computer_Integration_1st',
     'Human_and_Computer_Integration_2nd',
     'Practicum_Industry_Immersion_1st',
     'Practicum_Industry_Immersion_2nd',
     'Integrative_Programming_and_Tech_1st',
     'Integrative_Programming_and_Tech_2nd',
     'System_Integration_and_Architecture_1st',
     'System_Integration_and_Architecture_2nd',
     'Information_Assurance_and_Security_1_1st',
     'Information_Assurance_and_Security_1_2nd',
     'Information_Assurance_and_Security_2_1st',
     'Information_Assurance_and_Security_2_2nd',
     'Software_Engineering_1st',
     'Software_Engineering_2nd',
     'Networking_1_1st',
     'Networking_1_2nd',
     'Networking_2_1st',
     'Networking_2_2nd',
     'WebProg_1st',
     'WebProg_2nd'
]
CS_FEATURES = [
     'Sex',
     'Shiftee',
     'ComProg_1_1st',
     'ComProg_1_2nd',
     'ComProg_2_1st',
     'ComProg_2_2nd',
     'Intro_to_Computing_1st',
     'Intro_to_Computing_2nd',
     'Info_Management_1st',
     'Info_Management_2nd',
     'Operating_System_1st',
     'Operating_System_2nd',
     'Elective_1_1st',
     'Elective_1_2nd',
     'Elective_2_1st',
     'Elective_2_2nd',
     'Elective_3_1st',
     'Elective_3_2nd',
     'Data_Structures_1st',
     'Data_Structures_2nd',
     'Application_Dev_and_Emerging_Tech_1st',
     'Application_Dev_and_Emerging_Tech_2nd',
     'Human_and_Computer_Integration_1st',
     'Human_and_Computer_Integration_2nd',
     'Practicum_Industry_Immersion_1st',
     'Practicum_Industry_Immersion_2nd',
     'Digital_Design_1st',
     'Digital_Design_2nd',
     'Architecture_and_Organization_1st',
     'Architecture_and_Organization_2nd',
     'Programming_Languages_1st',
     'Programming_Languages_2nd',
     'Modelling_and_Simulation_1st',
     'Modelling_and_Simulation_2nd',
     'Information_Assurance_and_Security_1st',
     'Information_Assurance_and_Security_1_2nd',
     'Software_Engineering_1_1st',
     'Software_Engineering_1_2nd',
     'Software_Engineering_2_1st',
     'Software_Engineering_2_2nd',
     'Network_Management_1st',
     'Network_Management_2nd',
     'Advance_Database_1st',
     'Advance_Database_2nd',
     'WebProg_1st',
     'WebProg_2nd'
]

TARGET = 'Suggested_job_role'

Cat_Y = dataset[TARGET]
X_IT = dataset[IT_FEATURES]
X_CS = dataset[CS_FEATURES]

X_IT = X_IT.replace(np.nan, 0)
X_CS = X_CS.replace(np.nan, 0)

X_IT['Sex'] = ordinal_encoder.fit_transform(X_IT[['Sex']])
X_CS['Sex'] = ordinal_encoder.fit_transform(X_CS[['Sex']])
percent = "%"
#   CREATE FLASK APP
flask_app = Flask(__name__)
#   MAIN JOB ROLE MODEL FOR BOTH IT CS
model_IT = pickle.load(open(model_IT_path, "rb"))
model_CS = pickle.load(open(model_CS_path, "rb"))
#   TOP 5 COURSES SUGGESTION FOR BOTH IT CS
model_ITsuggest = pickle.load(open(model_ITsuggest_path, "rb"))
model_CSsuggest = pickle.load(open(model_CSsuggest_path, "rb"))

@flask_app.route("/")

def Home():
    return render_template("index.html")

# @flask_app.route("/faculty_end")
# def facultyEnd_view():
#     return render_template("Faculty/facultyEnd.html")

@_route_it.route('/login_register_IT', methods=['GET'])
def login_registerIT_view():
    auth_user=current_user
    if auth_user.is_authenticated:
        if auth_user.user_type == 1 and auth_user.department == "information technology":
            return redirect(url_for('.it_dashboard'))
        else:
            return redirect(url_for('_auth.index'))
    return render_template("IT/login_registerIT.html")

@_route_it.route('/login_IT', methods=['GET', 'POST'])
def login_IT():
    auth_user=current_user
    if auth_user.is_authenticated:
        if auth_user.user_type == 1 and auth_user.department == "information technology":
            return redirect(url_for('.it_dashboard'))
        else:
            return redirect(url_for('_auth.index'))
    else:
        if request.method == 'POST':
            user = User.query.filter_by(email=request.form['email'], department='information technology').first()
            if user:
                if check_password_hash(user.password, request.form['password']):
                    login_user(user, remember=True)
                    return redirect(url_for('.it_dashboard'))
                else:
                    flash('Invalid or wrong password', category='error')
            else:
                flash('No record found', category='error')
    return redirect(url_for('.login_registerIT_view'))

@_route_it.route('/signupIT', methods=['POST'])
def signupIT():
    try:
        new_user = User(request.form['first_name'], request.form['middle_name'], request.form['last_name'], request.form['contact_number'], request.form['email'], 'information technology', (generate_password_hash(request.form['password'], method="sha256")), 1)
        db.session.add(new_user)
        db.session.commit()
        flash('Account successfully created', category='success_register')
        return redirect(url_for('.login_registerIT_view'))
    except:
        flash('Invalid credentials', category='error')
        return redirect(url_for('.login_registerIT_view'))


@_route_it.route('/it_dashboard', methods=['GET'])
@login_required
def it_dashboard():
    auth_user=current_user
    if auth_user.user_type == 1 and auth_user.department == "information technology":
        return render_template("IT/ITinputs.html", auth_user=auth_user)
    else:
        return redirect(url_for('_auth.index'))
    
    # return render_template("IT/ITinputs.html", auth_user=auth_user)

@_route_it.route("/ITinputs_", methods=['GET'])
def predict_IT_():
    auth_user=current_user
    if auth_user.user_type == 1 and auth_user.department == "information technology":
        return render_template("IT/ITinputs.html")
    else:
        return redirect(url_for('_auth.index'))
    
@_route_it.route("/predict_IT", methods=["GET", "POST"])
def predict_IT():
    auth_user=current_user
    if request.method== 'GET':
        return render_template("IT/ITinputs.html")
    else:
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        new_Xdata_IT = X_IT.sample(4)
        new_Ydata_IT = Cat_Y[new_Xdata_IT.index.values]
        pred_IT = model_IT.predict(features)
        suggestIT = model_ITsuggest.predict(pred_IT)

        def recall(new_Ydata_IT, pred_IT, K):
                act_set = set(new_Ydata_IT)
                pred_set = set(pred_IT[:K])
                result = round(len(act_set & pred_set) / float(len(act_set)), 2)
                return result
            
        for K in range(0, 3):
            
            predIT = pred_IT[0]
            actual = new_Ydata_IT
            prediction = ["Software Engineer / Programmer", "Technical Support Specialist", "Academician", "Administrative Assistant"]
            random.shuffle(prediction)
            prediction = [prediction.replace(predIT, '0')for prediction in prediction]
            prediction.append(predIT)
            
            fetch1 = recall(actual, prediction, K)
            if fetch1 == 1.0:
                fetch1 = 100
            elif fetch1 == 0.75:
                fetch1 = 75
            elif fetch1 == 0.67:
                fetch1 = 67 
            elif fetch1 == 0.5:
                fetch1 = 50
            elif fetch1 == 0.33:
                fetch1 = 33
            elif fetch1 == 0.25:
                fetch1 = 25
            else:
                fetch1 = 0
                
            fetch2 = recall(actual, prediction, K-1)
            if fetch2 == 1.0:
                fetch2 = 100
            elif fetch2 == 0.75:
                fetch2 = 75
            elif fetch2 == 0.67:
                fetch2 = 67
            elif fetch2 == 0.5:
                fetch2 = 50
            elif fetch2 == 0.33:
                fetch2 = 33
            elif fetch2 == 0.25:
                fetch2 = 25
            else:
                fetch2 = 0
                
            fetch3 = recall(actual, prediction, K-2)
            if fetch3 == 1.0:
                fetch3 = 100
            elif fetch3 == 0.75:
                fetch3 = 75
            elif fetch3 == 0.67:
                fetch3 = 67
            elif fetch3 == 0.5:
                fetch3 = 50
            elif fetch3 == 0.33:
                fetch3 = 33
            elif fetch3 == 0.25:
                fetch3 = 25
            else:
                fetch3 = 0
                
            fetch4 = recall(actual, prediction, K-3)
            if fetch4 == 1.0:
                fetch4 = 100
            elif fetch4 == 0.75:
                fetch4 = 75
            elif fetch4 == 0.67:
                fetch4 = 67
            elif fetch4 == 0.5:
                fetch4 = 50
            elif fetch4 == 0.33:
                fetch4 = 33
            elif fetch4 == 0.25:
                fetch4 = 25
            else:
                fetch4 = 0
                
            fetchPred1 = prediction[0]
            fetchPred2 = prediction[-1]
            fetchPred3 = prediction[-2]
            fetchPred4 = prediction[-3]
            
            
            '''
                ADD NEW RECORD TO DATABASE
            '''
            
            pred2 = "No Result" if fetch2 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred2 == '0' else "{}".format(f"{prediction[K-1]} : {fetch2}%")
            pred3 = "No Result" if fetch3 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred3 == '0' else "{}".format(f"{prediction[K-2]} : {fetch3}%"),
            pred4 = "No Result" if fetch4 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred4 == '0' else "{}".format(f"{prediction[K-3]} : {fetch4}%")
            pred1 = "No Result" if fetch1 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred1 == '0' else "{}".format(f"{prediction[K]} : {fetch1}%")
            
            new_pred = PredictionResult(pred2, pred4, pred3, pred1, current_user.id)
            db.session.add(new_pred)
            db.session.commit()
            
            return render_template("/IT/ITPredRes.html", 
                                prediction_text1 = "" if fetch1 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred1 == '0' else "{}".format(f"{prediction[K]} : {fetch1}%"), 
                                prediction_text2 = "" if fetch2 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred2 == '0' else "{}".format(f"{prediction[K-1]} : {fetch2}%"),
                                prediction_text3 = "" if fetch3 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred3 == '0' else "{}".format(f"{prediction[K-2]} : {fetch3}%"),
                                prediction_text4 = "" if fetch4 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 == 0 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" or fetchPred4 == '0' else "{}".format(f"{prediction[K-3]} : {fetch4}%"),
                                prediction_label1 = "" if fetch2 == 0 or fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 == 1.0 and fetchPred2 == "Administrative Assistant" else "{}".format(f"{prediction[K-1]} is a more likely career path for you. Congratulations!"),
                                prediction_label3 = "{}".format(f"We apologize for the poor results caused by the anomaly our algorithm discovered when performing the prediction....") if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 else "",
                                prediction_label4 = "- In addition to the possibilities mentioned above, there is still a significant chance that you will be hired for your first job in one or more of the positions listed on the left side." if fetch1 > 0 and fetchPred1 !="0" or fetch3 > 0 and fetchPred3 !="0" or fetch4 > 0 and fetchPred4 !="0" or fetchPred2 == "Administrative Assistant" else "",
                                job_label1 = "{}".format(f"{prediction[K-1]} = {fetch2}%") if fetch2 <= 100 and fetch1 >= 0 and fetch3 >= 0 and fetch4 >= 0 and fetchPred2 == "Administrative Assistant" else "",
                                job_label2 = "Predicted IT/CS Related Job(s)" if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch1 == 0 and fetch2 <= 100 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" else "Predicted IT/CS Related Job(s)",
                                label_text2 = "- To increase the likelihood of landing IT/CS-related jobs, the below courses on the left side must be improved." if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 >= 0 and fetch3 >= 0 and fetch4 >= 0 and fetchPred2 == "Administrative Assistant" or fetch2 <= 100 and fetch1 >= 0 and fetch3 >= 0 and fetch4 >= 0 and fetchPred2 == '0' or fetch1 == 0 and fetch2 <= 100 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" else "",
                                course_suggestion2 = "{}".format(suggestIT.tolist()) if fetch1 == 0 and fetch2 == 0 and fetch3 == 0 and fetch4 == 0 or fetch2 <= 100 and fetch1 >= 0 and fetch3 >= 0 and fetch4 >= 0 and fetchPred2 == "Administrative Assistant" or fetch2 <= 100 and fetch1 >= 0 and fetch3 >= 0 and fetch4 >= 0 and fetchPred2 == '0' or fetch1 == 0 and fetch2 <= 100 and fetch3 == 0 and fetch4 == 0 and fetchPred2 == "Administrative Assistant" else "",
                                auth_user=auth_user
                                )