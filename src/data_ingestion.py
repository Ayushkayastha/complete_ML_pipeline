import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split


#making file for logs to be stored
log_dir='logs'
os.makedirs(log_dir, exist_ok=True)

#we make an obj of an logger
logger= logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

#for debug loggger we make an instance
console_handler= logging.StreamHandler()
console_handler.setLevel('DEBUG')

#for file loggger we make an instance
log_file_path= os.path.join(log_dir,'data_ingestion.log')
file_handler= logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

#set in which format logs shoud be shown
formater=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formater)
file_handler.setFormatter(formater)

# adding 2 handlers we created to logger obj
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(calorie_data_url: str,exercise_data_url: str) -> pd.DataFrame:
    try:
        df1=pd.read_csv(calorie_data_url)
        logger.debug('calorie data is loaded')

        df2=pd.read_csv(exercise_data_url)
        logger.debug('exercise data is loaded')

        df=pd.merge(df1,df2,how='inner',on='User_ID')
        return df
    
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise

    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise
def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)

    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
       calorie_data_url='https://raw.githubusercontent.com/Ayushkayastha/calorie_burn_dataset/refs/heads/main/calories.csv'
       exercise_data_url='https://raw.githubusercontent.com/Ayushkayastha/calorie_burn_dataset/refs/heads/main/exercise.csv'
       df= load_data(calorie_data_url=calorie_data_url,exercise_data_url=exercise_data_url)
       train_data, test_data = train_test_split(df, test_size=0.21, random_state=42)
       save_data(train_data, test_data, data_path='./data')


    except Exception as e:
        logger.error('Failed to complete data ingestion process: %s',e)

if __name__ == '__main__':
    main()