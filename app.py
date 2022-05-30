from flask import Flask, request, jsonify
#from tensorflow import keras
import joblib
import pandas as pd
from datetime import datetime
import os

from regex import B 

a = 2
b = 3
c = b 
print(a)
prefix = '/opt/ml/'
model_path = prefix

#base_dir = os.path.dirname(os.path.realpath(''))
#constants_abs_path = os.path.join(base_dir,r'newavm/models')
#model_path = constants_abs_path

app=Flask(__name__)

@app.route('/')
def welcome():
    return ""

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predciting the price of a given appartment.
    input:
        json file containing the following keys :
            address: which contains a dictionary with two keys: lat and lon.
            dataValues: which contains a dictionary with the following keys: 
                builyear, usableArea, primærrom, floorNumber, bedrooms, propertyNumber: str
                heis: str, value Heis. Should not exist in the dictionary if there is no heis
                takterrasse: str, none, Should not exist in the dictionary if there is no terrasse or balkony
                garasjeplass: str, value Garasjeplass. Should not exist in the dictionary if there is no heis
        i.e.: {"address":{"lat":59.913486,"lon":10.723724},
                "dataValues":{"heis":"Heis","garasjeplass":"Garasjeplass","propertyNumber":"H0102","floorNumber":"1","bedrooms":"1","primærrom":"47","usableArea":"42","builtYear":"1998"}}
    Response:
        dict['prediction'] contains the prediction: float
    '''

    models = []
    nameOfModels = ['LGBMRegressor', 'XGBRegressor', 'RandomForestRegressor', 'KNeighborsRegressor']
    for nameOfFileModel in nameOfModels:
        filename = os.path.join(constants_abs_path,"{}.pkl".format(nameOfFileModel))
        models.append(joblib.load(open(filename, 'rb')))
    modelStaking = joblib.load(os.path.join(model_path, 'StackingModel.pkl'))
    model_pipeline = joblib.load(os.path.join(model_path, 'model_pipeline.pkl'))
    modelNN= keras.models.load_model(os.path.join(model_path))
    appartment = {}
    data=request.get_json()
    if(set(['garasjeplass','parkeringute']).issubset( data['properties'].keys())):
        appartment['Parking'] = 1
    else:
        appartment['Parking'] = 0
    if(set(['takterrasse','vestbalkong','annenbalkong', 'fellestakerrasse']).issubset( data['properties'].keys())):
        appartment['F_BalkongTerrasse'] = 1
    else:
        appartment['F_BalkongTerrasse'] = 0
    if('heis' in data['properties'].keys()):
        appartment['F_Heis'] = 1
    else:
        appartment['F_Heis'] = 0
    appartment['buildyear'] = int(data['properties']['builtYear']["value"])
    appartment['bedrooms'] = int(data['properties']['bedrooms']["value"])
    appartment['floor'] = int(data['properties']['floorNumber']["value"])
    appartment['lng'] = float(data['address']['lon'])
    appartment['lat'] = float(data['address']['lat'])
    appartment['PROM'] = int(data['properties']['primærrom']["value"])
    appartment['BRA'] = int(data['properties']['usableArea']["value"])
    appartment['apartmentnumber'] = data['properties']['propertyNumber']["value"]
    appartment['sold_date_est'] = str(datetime.now())
    appartment = pd.DataFrame(appartment, index = [0])
    modelDictionary = {}
    for i in range(0, len(nameOfModels)):
        modelDictionary[nameOfModels[i]] = models[i]
    prediction=model_pipeline.predict(appartment,models=modelDictionary, model_NN=modelNN, stackingModel = modelStaking)
    Prom = int(appartment["PROM"].values)
    stackingPrediction = prediction[0]*Prom
    return jsonify({"prediction":stackingPrediction, "submodels": [{"model":prediction[2][i],"value":prediction[1][i]*Prom} for i in range(0, len(prediction[1]))]}), 200

if __name__=='__main__':
    app.run(host='0.0.0.0')