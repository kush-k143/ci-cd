import pickle
import json
import numpy as np
from azureml.core.model import Model
import joblib
import json


def init():
    global model

    # load the model from file into a global object
    model_path = Model.get_model_path(model_name="random_forest_regression_model.pkl")
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        Year = int(data['Year'])
        Present_Price=float(data['Present_Price'])
        Kms_Driven=int(data['Kms_Driven'])
        Kms_Driven2=np.log(Kms_Driven)
        Owner=int(data['Owner'])
        Fuel_Type_Petrol=data['Fuel_Type_Petrol']
        if(Fuel_Type_Petrol=='Petrol'):
                Fuel_Type_Petrol=1
                Fuel_Type_Diesel=0
        else:
            Fuel_Type_Petrol=0
            Fuel_Type_Diesel=1
        Year=2020-Year
        Seller_Type_Individual=data['Seller_Type_Individual']
        if(Seller_Type_Individual=='Individual'):
            Seller_Type_Individual=1
        else:
            Seller_Type_Individual=0    
        Transmission_Mannual=data['Transmission_Mannual']
        if(Transmission_Mannual=='Mannual'):
            Transmission_Mannual=1
        else:
            Transmission_Mannual=0
        prediction=model.predict([[Present_Price,Kms_Driven2,Owner,Year,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Mannual]])
        output=round(prediction[0],2)
        if output<0:
            return json.dumps(list({"Sorry you cannot sell this car"}))
        else:
            return json.dumps(list({"You Can Sell The Car at {}".format(output)}))
    except Exception as e:
        result = str(e)
        return json.dumps({"An error occured while prediction in Score.py file......": result})