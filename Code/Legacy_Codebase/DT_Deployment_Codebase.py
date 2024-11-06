import pandas as pd
import numpy as np
import pickle
import streamlit as st
# from PIL import image
import joblib

# start with DT model

pickle_in=open('../../Model/DT_Model_Deploy.pkl', 'rb')
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

def prediction(age, balance, duration, campaign, pdays, poutcome, days_in_year):
    prediction=classifier.predict([[age, balance, duration, campaign, pdays, poutcome, days_in_year]])
    print(prediction)
    return prediction
  
# this is the main function in which we define our webpage  
def main(): 
      # giving the webpage a title 
    # st.title("Term Deposit Subscription Prediction") 
      
    # here we define some of the front end elements of the web page like  
    # the font and background color, the padding and the text to be displayed 
    html_temp = """
    <div style="background-color:yellow; padding:13px;">
        <h1 style="color:black; text-align:center;">Term Deposit Subscription Prediction ML App (Decision Tree)</h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have  
    # defined in the above code 
    st.markdown(html_temp, unsafe_allow_html = True) 

    # User input fields for required features
    st.subheader("Please enter the following information:")
      
    # the following lines create text boxes in which the user can enter  
    # the data required to make the prediction 
    # age = st.text_input("Age (18-65):", "Type Here") 
    # age = st.number_input("Age (18-65):", min_value=18, max_value=65) 
    age = st.slider("Age (18-65):", min_value=18, max_value=65, value=42) 
    # education = st.text_input("Education:", "Type Here") 
    # default = st.text_input("Default:", "Type Here") 
    # balance = st.text_input("Bank account balance:", "Type Here")
    balance = st.number_input("Bank account balance (0-100000000):", min_value=0, max_value=100000000, value=10000)
    # housing = st.text_input("Housing:", "Type Here")
    # loan = st.text_input("Loan:", "Type Here")
    # duration = st.text_input("Duration of your last campaign (in hours)", "Type Here")
    duration = st.slider("Duration of your last campaign (in minutes)", min_value=0, max_value=300, value=15)

    campaign = st.slider("How many times did we contact you?", min_value=0, max_value=15, value=0)
    pdays = st.slider("How many days ago was your last contacted by us?", min_value=0, max_value=1000, value=0)
    # previous = st.text_input("Previous:", "Type Here")
    # poutcome = st.text_input("Poutcome:", "Type Here")

    # pdays = st.selectbox("Pdays:", [0, 1], index=0)  # Default value is 0
    poutcome = st.selectbox("Outcome of your last campaign?", ["Failure", "Unknown", "Success"], index=1)  # Default value is 0

    print(poutcome)
    if poutcome=="Failure":
        poutcome=0
    elif poutcome=="Unknown":
        poutcome=0.5
    elif poutcome=="Success":
        poutcome=1


    # contact_cellular = st.text_input("Contact Cellular:", "Type Here")
    # contact_telephone = st.text_input("Contact Telephone:", "Type Here")
    # marital_divorced = st.text_input("Marital Divorced:", "Type Here")
    # marital_married = st.text_input("Marital Married:", "Type Here")
    # marital_single = st.text_input("Marital Single:", "Type Here")
    # job_admin = st.text_input("Job Admin:", "Type Here")
    # job_blue_collar = st.text_input("Job Blue Collar:", "Type Here")
    # job_entrepreneur = st.text_input("Job Entrepreneur:", "Type Here")
    # job_housemaid = st.text_input("Job Housemaid:", "Type Here")
    # job_management = st.text_input("Job Management:", "Type Here")
    # job_retired = st.text_input("Job Retired:", "Type Here")
    # job_self_employed = st.text_input("Job Self Employed:", "Type Here")
    # job_services = st.text_input("Job Services:", "Type Here")
    # job_student = st.text_input("Job Student:", "Type Here")
    # job_technician = st.text_input("Job Technician:", "Type Here")
    # job_unemployed = st.text_input("Job Unemployed:", "Type Here")
    # job_unknown = st.text_input("Job Unknown:", "Type Here")
    days_in_year = st.slider("Number of Days in a Year:", min_value=0, max_value=365, value=150)
    

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
    
    # Predict button to trigger the prediction
    if st.button("Predict"):
        # Ensure all inputs are numeric where appropriate
        try:
            age = float(age)
            balance = float(balance)
            duration = float(duration)
            duration*=60
            print(duration)
            campaign = float(campaign)
            days_in_year = float(days_in_year)

            # Make the prediction using the input values
            result = prediction(age, balance, duration, campaign, pdays, poutcome, days_in_year)

            # Display success or failure based on the prediction result
            if result[0] == 1:
                msg = 'The marketing campaign will: \nSucceed! (Output=1)'
                st.success(msg)
            elif result[0] == 0:
                msg = 'The marketing campaign will: \nFail! (Output=0)'
                st.success(msg)

        except ValueError:
            st.error("Please enter valid numeric values for all fields!")

    # if st.button("Predict"):
    #     result=prediction(age, balance, duration, campaign, pdays, poutcome, days_in_year)
    
    # # print(result[0])
    # # print(type(result))
    #     if result[0]== 1:
    #         msg='The marketing campaign will:\nsucceed! (Output=1)'
    #         st.success(msg)  
    #     elif result[0]== 0:
    #         msg='The marketing campaign will:\nfail! (Output=0)' 
    #         st.success(msg) 

    # st.success(msg)  

    # if need to see other inputs
    # st.success('The output is {}'.format(result)) 
     
if __name__=='__main__': 
    main() 