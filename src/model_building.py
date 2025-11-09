import os
import numpy as np
import pandas as pd
import pickle
import logging
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import yaml

#making file for logs to be stored
log_dir='logs'
os.makedirs(log_dir, exist_ok=True)

#we make an obj of an logger
logger= logging.getLogger('model_building')
logger.setLevel('DEBUG')

#for debug loggger we make an instance
console_handler= logging.StreamHandler()
console_handler.setLevel('DEBUG')

#for file loggger we make an instance
log_file_path= os.path.join(log_dir,'model_building.log')
file_handler= logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

#set in which format logs shoud be shown
formater=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formater)
file_handler.setFormatter(formater)

# adding 2 handlers we created to logger obj
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)#converts yaml content to dictionary
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading data: %s', e)
        raise

def train_model(X_train:pd.DataFrame,y_train:pd.DataFrame,params:dict)->XGBRegressor:
    try:
      #  model=XGBRegressor(
            #colsample_bytree= params['colsample_bytree'],
            #max_depth= params['max_depth'], 
          #  n_estimators= params['n_estimators'],
         #   subsample= params['subsample'],
        #)
        model=XGBRegressor()
        logger.debug('Model started training')
        model.fit(X_train,y_train)
        logger.debug('Model training completed')
        return model
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model:XGBRegressor,path:str)->None:
    try:
        os.makedirs(os.path.dirname(path),exist_ok=True)
        with open(path, 'wb') as file:
            pickle.dump(model, file)

        logger.debug("Model saved successfully at %s",path)

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def main():
    try:
        params=load_params(params_path='params.yaml')
        colsample_bytree=params['model_building']['colsample_bytree']
        learning_rate=params['model_building']['learning_rate']
        max_depth=params['model_building']['max_depth']
        n_estimators=params['model_building']['n_estimators']
        subsample=params['model_building']['subsample']
        training_data=load_data(file_path='./data/processed/train_processed.csv')
        X_train = training_data.iloc[:, :-1].values
        y_train = training_data.iloc[:, -1].values
        params={
            'colsample_bytree': colsample_bytree, 
            'learning_rate': learning_rate, 
            'max_depth': max_depth, 
            'n_estimators': n_estimators, 
            'subsample': subsample
            }
        trained_model=train_model(X_train=X_train,y_train=y_train,params=params)
        model_path='models/model.pkl'
        save_model(model=trained_model,path=model_path)

    except Exception as e:
        logger.error('Failed to complete the model_building: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()   