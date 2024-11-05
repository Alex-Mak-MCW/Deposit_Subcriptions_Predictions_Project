import pandas as pd
import numpy as np
import pickle
import streamlit as st
import joblib

# RF model

pickle_in=open('../../Model/XGB_Model_Deploy.pkl', 'rb')
# classifier=pickle.load(pickle_in)
pipeline=pickle.load(pickle_in)
classifier=pipeline.named_steps['classifier']

# pickle_in=open('../Codebase/test_DT_Model.pkl', 'rb')
# pipeline=pickle.load(pickle_in)
# classifier=pipeline.named_steps['classifier']

def welcome():
    return 'HiHi!'

# defining the function which will make the prediction using  
# the data which the user inputs 

# ['age', 'education', 'default', 'balance', 'housing', 'loan',
#        'duration', 'campaign', 'pdays', 'previous', 'poutcome',
#        'contact_cellular', 'contact_telephone', 'marital_divorced',
#        'marital_married', 'marital_single', 'job_admin.',
#        'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
#        'job_management', 'job_retired', 'job_self-employed',
#        'job_services', 'job_student', 'job_technician', 'job_unemployed',
#        'job_unknown', 'days_in_year']


# def prediction(age, education, default, balance, housing, loan,
#        duration, campaign, pdays, previous, poutcome,
#        contact_cellular, contact_telephone, marital_divorced,
#        marital_married, marital_single, job_admin,
#        job_blue_collar, job_entrepreneur, job_housemaid,
#        job_management, job_retired, job_self_employed,
#        job_services, job_student, job_technician, job_unemployed,
#        job_unknown, days_in_year):   
   
#     prediction = classifier.predict( 
#         [[age, education, default, balance, housing, loan,
#        duration, campaign, pdays, previous, poutcome,
#        contact_cellular, contact_telephone, marital_divorced,
#        marital_married, marital_single, job_admin,
#        job_blue_collar, job_entrepreneur, job_housemaid,
#        job_management, job_retired, job_self_employed,
#        job_services, job_student, job_technician, job_unemployed,
#        job_unknown, days_in_year]]) 
#     print(prediction) 
#     return prediction 

def prediction(housing, loan, duration, pdays, poutcome, marital_married, job_blue_collar, job_housemaid, days_in_year):
    # prediction=classifier.predict([[age, balance, duration, campaign, pdays, poutcome, days_in_year]])
    prediction=classifier.predict([[housing, loan, duration, pdays, poutcome, marital_married, job_blue_collar, job_housemaid, days_in_year]])
    print(prediction)
    return prediction
  
# this is the main function in which we define our webpage  
def main(): 
      # giving the webpage a title 
    # st.title("Term Deposit Subscription Prediction") 
      
    # here we define some of the front end elements of the web page like  
    # the font and background color, the padding and the text to be displayed 
    html_temp = """ 
    <div style ="background-color:red;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Term Deposit Subscription Prediction ML App (XGBoost)</h1> 
    </div> 
    """
      
    # this line allows us to display the front end aspects we have  
    # defined in the above code 
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # the following lines create text boxes in which the user can enter  
    # the data required to make the prediction 
    # age = st.text_input("Age:", "Type Here") 
    # education = st.text_input("Education:", "Type Here") 
    # default = st.text_input("Default:", "Type Here") 
    # balance = st.text_input("Balance:", "Type Here") 
    housing = float(st.text_input("Housing:", 0))
    loan = float(st.text_input("Loan:", 0))
    duration = float(st.text_input("Duration:", 0))
    # campaign = st.text_input("Campaign:", "Type Here")
    pdays = float(st.text_input("Pdays:", 0))
    # previous = st.text_input("Previous:", "Type Here")
    poutcome = float(st.text_input("Poutcome:", 0))
    # contact_cellular = st.text_input("Contact Cellular:", "Type Here")
    # contact_telephone = st.text_input("Contact Telephone:", "Type Here")
    # marital_divorced = st.text_input("Marital Divorced:", "Type Here")
    marital_married = float(st.text_input("Marital Married:", 0))
    # marital_single = st.text_input("Marital Single:", "Type Here")
    # job_admin = st.text_input("Job Admin:", "Type Here")
    job_blue_collar = float(st.text_input("Job Blue Collar:", 0))
    # job_entrepreneur = st.text_input("Job Entrepreneur:", "Type Here")
    job_housemaid = float(st.text_input("Job Housemaid:", 0))
    # job_management = st.text_input("Job Management:", "Type Here")
    # job_retired = st.text_input("Job Retired:", "Type Here")
    # job_self_employed = st.text_input("Job Self Employed:", "Type Here")
    # job_services = st.text_input("Job Services:", "Type Here")
    # job_student = st.text_input("Job Student:", "Type Here")
    # job_technician = st.text_input("Job Technician:", "Type Here")
    # job_unemployed = st.text_input("Job Unemployed:", "Type Here")
    # job_unknown = st.text_input("Job Unknown:", "Type Here")
    days_in_year = float(st.text_input("Days in Year:", 0))
    

    result ="" 
      
    # the below line ensures that when the button called 'Predict' is clicked,  
    # the prediction function defined above is called to make the prediction  
    # and store it in the variable result 
    # if st.button("Predict"): 
    #     result = prediction(age, education, default, balance, housing, loan,
    #    duration, campaign, pdays, previous, poutcome,
    #    contact_cellular, contact_telephone, marital_divorced,
    #    marital_married, marital_single, job_admin,
    #    job_blue_collar, job_entrepreneur, job_housemaid,
    #    job_management, job_retired, job_self_employed,
    #    job_services, job_student, job_technician, job_unemployed,
    #    job_unknown, days_in_year) 

    if st.button("Predict"):
        # result=prediction(age, balance, duration, campaign, pdays, poutcome, days_in_year)
        result=prediction(housing, loan, duration, pdays, poutcome, marital_married, job_blue_collar, job_housemaid, days_in_year)
    
    # print(result[0])
    # print(type(result))
        if result[0]== 1:
            msg='The marketing campaign will:\nsucceed! (Output=1)'
            st.success(msg)  
        elif result[0]== 0:
            msg='The marketing campaign will:\nfail! (Output=0)' 
            st.success(msg) 

    # st.success(msg)  

    # if need to see other inputs
    # st.success('The output is {}'.format(result)) 
     
if __name__=='__main__': 
    main() 