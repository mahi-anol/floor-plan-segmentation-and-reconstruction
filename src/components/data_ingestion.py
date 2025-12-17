import os
import shutil
from dataclasses import dataclass
from src.logger import logging
from tqdm import tqdm
from src.exception import CustomException
import sys
import random

@dataclass
class  DataIngestionConfig:
    data_source: str = "./data-source/dataset2/ImagesGT"
    raw_train_data_location:str="./artifacts/raw-data/train"  # The location where the ingested raw data will be saved.
    raw_test_data_location:str="./artifacts/raw-data/test"  # The location where the ingested raw data will be saved.
    train_test_ratio:int=0.8 

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):

        try:
            logging.info("Initiating Data Ingestion...")
            files=os.listdir(self.ingestion_config.data_source)

            svg_files=[]
            image_files=[]

            for file in tqdm(files,desc="listing valid files",leave=False):
                if file.endswith('.svg'):
                    svg_files.append(file)
                elif file.endswith('.png') or file.endswith('.jpg'):
                    image_files.append(file)

            svg_image_pair=[]
            for svg_file in tqdm(svg_files,desc="separating image vs mask pair",leave=False):
                parts=str(svg_file).split('_gt')
                if parts[0]+'.png' in image_files:
                    svg_image_pair.append((svg_file,parts[0]+'.png'))
                elif parts[0]+'.jpg' in image_files:
                    svg_image_pair.append((svg_file,parts[0]+'.jpg'))
                else:
                    print(svg_file)
            
            ### shuffling to introduce some randomness in the train and test split.
            random.Random(42).shuffle(svg_image_pair)

            train_data_length=len(svg_image_pair)*self.ingestion_config.train_test_ratio

            for i,(svg,image) in enumerate(tqdm(svg_image_pair,desc="copying pairs to desired location",leave=False)):
                svg_path=os.path.join(self.ingestion_config.data_source,svg)
                image_path=os.path.join(self.ingestion_config.data_source,image)

                if i<train_data_length:
                    copy_location=self.ingestion_config.raw_train_data_location+f'/data-{i}'
                else:
                    copy_location=self.ingestion_config.raw_test_data_location+f'/data-{i}'

                os.makedirs(copy_location,exist_ok=True)
                shutil.copy(svg_path,copy_location)
                shutil.copy(image_path,copy_location)
            
            logging.info("Data ingestion Finished")
            return self.ingestion_config.raw_train_data_location, self.ingestion_config.raw_test_data_location
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    train_data_path,test_data_path=DataIngestion().initiate_data_ingestion()
    print("Train data path: ",train_data_path)
    print("Test data path: ",test_data_path)
