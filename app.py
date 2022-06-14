"""
app.py
"""
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('index1.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Age = int(request.form['Age'])
        Gendar = (request.form['Gendar'])
        if(Gendar == 'Female'):
            Gendar = 0
        else:
            Gendar = 1
        Marital_Status = (request.form['Marital_Status'])
        if Marital_Status == 'Single':
            Marital_Status = 2
        elif(Marital_Status == 'Divorced'):
            Marital_Status = 0
        else:
            Marital_Status = 1
        education = (request.form['education'])
        if(education == 'Bachelor'):
            education = 0
        elif(education == 'Master'):
            education = 1
        elif(education == 'Phd'):
            education = 2
        else:
            education = 3
        Environmentsat = (request.form['Environmentsat'])
        if(Environmentsat == 'high'):
            Environmentsat = 0
        elif(Environmentsat == 'low'):
            Environmentsat = 1
        elif(Environmentsat == 'medium'):
            Environmentsat = 2
        else:
            Environmentsat = 3
        job_involvement = (request.form['job_involvement'])
        if(job_involvement == 'high'):
            job_involvement = 0
        elif(job_involvement == 'low'):
            job_involvement = 1
        elif(job_involvement == 'medium'):
            job_involvement = 2
        else:
            job_involvement = 3
        job_level = int(request.form['job_level'])
        job_satisfaction = (request.form['job_satisfaction'])
        if(job_satisfaction == 'high'):
            job_satisfaction= 0
        elif(job_satisfaction == 'low'):
            job_satisfaction = 1
        elif(job_satisfaction == 'medium'):
            job_satisfaction = 2
        else:
            job_satisfaction = 3
        annual_income = int(request.form['annual_income'])
        relationship_sat = (request.form['relationship_sat'])
        if(relationship_sat == 'high'):
            relationship_sat= 0
        elif(relationship_sat == 'low'):
            relationship_sat = 1
        elif(relationship_sat == 'medium'):
            relationship_sat = 2
        else:
            relationship_sat = 3
        working_hrs = (request.form['working_hrs'])
        if (working_hrs == 'greaterthan9'):
            working_hrs = 1
        elif (working_hrs == 'equalto9'):
            working_hrs = 0
        else:
            working_hrs = 2
        experience = int(request.form['experience'])
        training_time = int(request.form['training_time'])

        worklife_balance = (request.form['worklife_balance'])
        if(worklife_balance == 'bad'):
            worklife_balance = 0
        elif(worklife_balance == 'best'):
            worklife_balance = 1
        elif(worklife_balance == 'better'):
            worklife_balance = 2
        else :
            worklife_balance = 3
        behaviourcompetence = (request.form['behaviourcompetence'])
        if(behaviourcompetence == 'excellent'):
            behaviourcompetence = 0
        elif(behaviourcompetence == 'inadequate'):
            behaviourcompetence = 1
        elif(behaviourcompetence == 'poor'):
            behaviourcompetence = 2
        elif(behaviourcompetence =='satisfactory'):
            behaviourcompetence = 3
        elif(behaviourcompetence =='very_good'):
            behaviourcompetence = 4
        ontime_delivery = (request.form['ontime_delivery'])
        if(ontime_delivery == 'excellent'):
            ontime_delivery = 0
        elif(ontime_delivery == 'good'):
            ontime_delivery = 1
        elif(ontime_delivery == 'poor'):
            ontime_delivery = 2
        else:
            ontime_delivery = 3
        ticket_solving_management = (request.form['ticket_solving_management'])
        if(ticket_solving_management == 'excellent'):
            ticket_solving_management = 0
        elif(ticket_solving_management == 'good'):
            ticket_solving_management = 1
        elif(ticket_solving_management == 'poor'):
            ticket_solving_management = 2
        else:
            ticket_solving_management = 3
        project_completion = int(request.form['project_completion'])
        working_from_home = (request.form['working_from_home'])
        if(working_from_home == 'no'):
            working_from_home = 0
        else:
            working_from_home = 1
        psycho_social_indicator = (request.form['psycho_social_indicator'])
        if(psycho_social_indicator == 'excellent'):
            psycho_social_indicator = 0
        elif(psycho_social_indicator == 'inadequate'):
            psycho_social_indicator = 1
        elif(psycho_social_indicator == 'poor'):
            psycho_social_indicator = 2
        elif(psycho_social_indicator =='satisfactory'):
            psycho_social_indicator = 3
        elif(psycho_social_indicator =='very_good'):
            psycho_social_indicator = 4
        over_time = (request.form['over_time'])
        if(over_time == 'no'):
            over_time = 0
        else:
            over_time = 1
        attendance = (request.form['attendance'])
        if(attendance == 'good'):
            attendance = 0
        else:
            attendance = 1
        percent_salary_hike = int(request.form['percent_salary_hike'])
        net_connection = (request.form['net_connection'])
        if(net_connection == 'good'):
            net_connection = 0
        else:
            net_connection = 1
        department = (request.form['department'])
        if(department == 'finance'):
            department = 0
        elif(department == 'HRM'):
            department = 1
        elif(department == 'IT'):
            department = 2
        elif(department =='RD'):
            department = 3
        elif(department =='sales'):
            department = 4
        position = (request.form['position'])
        if(position == 'analyst'):
            position = 0
        elif(position == 'developer'):
            position = 1
        elif(position == 'executive'):
            position = 2
        elif(position =='HR'):
            position = 3
        elif(position =='manager'):
            position = 4
        elif position =='scientist':
            position = 5
        elif position == 'teamleader':
            position = 6

        prediction1 = model.predict([[Age, Gendar, Marital_Status, education, Environmentsat, job_involvement, job_level, job_satisfaction, annual_income, relationship_sat, working_hrs, experience, training_time, worklife_balance, behaviourcompetence, ontime_delivery, ticket_solving_management, project_completion, working_from_home, psycho_social_indicator, over_time, attendance, percent_salary_hike, net_connection, department, position]])
        #output = round(prediction[0], 2)
        if prediction1 < 0:
            return render_template('index2.html', prediction_texts="you did something wrong")
        elif prediction1 == 1:
            return render_template('index2.html', prediction_text="Performance Rating for  Employee is good \n")
        elif prediction1 == 2:
            return render_template('index2.html', prediction_text="Performance Rating for Employee is poor\n")
        elif prediction1 == 0:
            return render_template('index2.html', prediction_text="Performance Rating for Employee is average \n")
    else:
        return render_template('index2.html')


if __name__ == "__main__":
    app.run(debug=True)
