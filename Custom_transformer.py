from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from datetime import datetime
import os
import pandas as pd
import numpy as np

def pre_processingData(X):
    '''
    Used for data preprocessing: adding, deleting and transforming features.
    Args: 
        data
    Output:
        transformed data
    '''
    t0 = datetime.strptime('2014-01-01 00:00:00', "%Y-%m-%d %H:%M:%S")

    X=X.copy()
    X['sold_date_est'] = pd.to_datetime(X.sold_date_est).dt.tz_localize(None)
    X['antype'] = X['apartmentnumber'].fillna(0).astype(str).apply( lambda x: 0*(x[0] =='H')+1*(x[0] =='L')+2*(x[0] =='U')+3*(x[0] =='K'))
    delta = X['sold_date_est'] - t0
    X.insert(0, 'days', delta.dt.days)
    S_age = X['sold_date_est'].apply(lambda x: x.year) - X['buildyear']
    X.insert(0, 'age', S_age)
    X["SROM"] = X["BRA"] - X["PROM"]
    X['division']=X['PROM']/(X['bedrooms'].replace(0,1))

    X["Accessibility"]  = 1
    X.loc[X[X["floor"]>1][X["F_Heis"]==0].index, ["Accessibility"]] = 0

    X = X.drop(['apartmentnumber','sold_date_est', "buildyear", "BRA"],axis=1)
    return X

class PreprocessData(BaseEstimator):
    '''
    Custom transformer for data preprocessing for pipeline
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pre_processingData(X)

class applyPCA(BaseEstimator):
    '''
    Custom transformer for pca for pipeline
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pca = PCA(0.95)
        pca = pca.fit(X.iloc[:, -205:])
        self.pca = pca
        return self

    def transform(self, X):
        pca =  self.pca
        dataPCA = pca.transform(X.iloc[:, -205:])
        transformed_PCA = pd.DataFrame(dataPCA)
        partOfX = (X.iloc[:, :14])
        transformed_X = transformed_PCA.join(partOfX)
        return transformed_X
    
class Model(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def predict(self, X, models=None, model_NN= None, stackingModel=None):
        self.models = list(models.values())
        self.nameModels = list(models.keys())
        self.model_NN = model_NN
        predictions = []
        for i in range(0, len(models)):
            predictions.append(self.models[i].predict(X)[0])
        prediction_NN = self.model_NN.predict(X)
        predictions.append(prediction_NN[0][0])
        self.nameModels.append("NN")
        print(predictions)
        columns = self.nameModels
        pred_df = pd.DataFrame([predictions], columns=columns)
        predictionFromStackingModel =  np.expm1(stackingModel.predict(pred_df))

        return [predictionFromStackingModel[0], np.expm1(predictions), self.nameModels]

def coords2distance(X0,X1) :
    # X0: scalars [lng,lat].
    # X1: vectors [lng,lat], 
    R = 6373.0    # radius of the Earth

    lng0 = np.radians(X0[0])
    lat0 = np.radians(X0[1])
    lng1 = np.radians(X1[:,0])
    lat1 = np.radians(X1[:,1])
    
    dlng = lng1 - lng0
    dlat = lat1 - lat0
    a = np.sin(dlat/2)**2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlng/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c *1000  # Distance in meters
    return distance
   
def create_location_points(df,dist_m) :
    ind = df.reset_index().index.to_list()
    X = df[['lng','lat']].to_numpy()
    lng_l = []
    lat_l = []


    while len(ind) > 0 :
        # Pick location index and find neaby locations
        ii = ind.pop()
        i_d = np.where(coords2distance(X[ii],X[ind] ) < dist_m )[0]
        i_d = [ind[i] for i in i_d ]
        
        # Remove nearby locations and add average to output list
        Ni = len(i_d)
        lng_l.append( (np.sum(X[i_d,0])+X[ii,0])/(Ni+1) )
        lat_l.append( (np.sum(X[i_d,1])+X[ii,1])/(Ni+1) )
        for i in i_d :
            ind.remove(i)

    return lng_l, lat_l


def distance_feature_transform(dist):
    return np.maximum(2-dist,0)


def add_geofeatures(df, df_ds, dist_m):

    df_geo = df.copy()
    features_geo = list(df.columns)
    for i in range( len(df_ds) ):
        feature_name = 'dist'+str(i)
        
        dist = coords2distance(df_ds.iloc[i].values, df[['lng','lat']].values)
        df_geo = df_geo.copy()
        df_geo[feature_name] = distance_feature_transform(dist/dist_m)

        features_geo.append(feature_name)
    
    return df_geo, features_geo

def CreateLocationPoints(df_sel, dist_m):
    
    df_ds = df_sel[['lng','lat']]
    for i in range(5) :
        lng_l, lat_l = create_location_points(df_ds,dist_m)
        df_ds = pd.DataFrame({'lng':lng_l, 'lat':lat_l,})
    return df_ds
    

class GeoFeatures(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, df_sel, y=None):
        dist_m = 500
        df_ds = CreateLocationPoints(df_sel,dist_m)
        self.df_ds=df_ds
        return self

    def transform(self, X):
        dist_m = 500
        X_dataset_geo, features_geo = add_geofeatures(X, self.df_ds, dist_m)
        return X_dataset_geo
