import os
import logging
import pandas as pd
import numpy as np


#making file for logs to be stored
log_dir='logs'
os.makedirs(log_dir, exist_ok=True)

#we make an obj of an logger
logger= logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

#for debug loggger we make an instance
console_handler= logging.StreamHandler()
console_handler.setLevel('DEBUG')

#for file loggger we make an instance
log_file_path= os.path.join(log_dir,'data_preprocessing.log')
file_handler= logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

#set in which format logs shoud be shown
formater=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formater)
file_handler.setFormatter(formater)

# adding 2 handlers we created to logger obj
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def drop_columns(df:pd.DataFrame,column_names:list)->pd.DataFrame:
    try:
       df=df.drop(column_names,axis=1) 
       logger.debug('Extra columns dropped')
       return df
    except Exception as e:
        logger.error('Error during dropping column: %s', e)
        raise

def map_gender(df:pd.DataFrame)->pd.DataFrame:
    try:
       df["Gender"]=df["Gender"].map({'male':0,'female':1})
       logger.debug('Gender column mapped')
       return df
    except Exception as e:
        logger.error('Error during maping gender: %s', e)
        raise

def remove_outliers(df:pd.DataFrame,features:list)->pd.DataFrame:
    try:
       for col in features:
           Q1,Q3=np.percentile(df[col],[25,75])        
           IQR=Q3-Q1
           lower_fence= Q1-1.5*IQR
           higher_fence= Q3+1.5*IQR
           
           df_clean = df[(df[col] >= lower_fence) & (df[col] <= higher_fence)]
       
       logger.debug('Outliers removed')
       return df_clean
    
    except Exception as e:
        logger.error('Error removing outliers: %s', e)
        raise

def preprocess_df(df:pd.DataFrame)->pd.DataFrame:
    df=drop_columns(df=df,column_names=['Body_Temp','Height','Weight','User_ID'])
    df=map_gender(df=df)
    df=remove_outliers(df=df,features=['Heart_Rate'])

    return df


def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
        raw_data_path=os.path.join(data_path,'processed')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train_processed.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test_processed.csv'),index=False)
        logger.debug('Processed Train and test data saved to %s', raw_data_path)

    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv('./data/raw/test.csv')

        train_processed_data=preprocess_df(df=train_data)
        test_processed_data=preprocess_df(df=test_data)

        save_data(train_data=train_processed_data,test_data=test_processed_data,data_path='./data')

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data preprocessing: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()   