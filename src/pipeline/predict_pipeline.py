import sys 
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipelines:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            predict=model.predict(data_scaled)
            return predict
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:

    def __init__(self,
                 gender:str,
                 ethnicity:str,
                 parential_level_of_education,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score:int,
                 writing_score:int):
        
        
        self.gender=gender
        self.ethnicity=ethnicity
        self.parential_level_of_education=parential_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                "gender":[self.gender],
                "race/ethnicity":[self.ethnicity],
                "parental level of education":[self.parential_level_of_education],
                "lunch":[self.lunch],
                "test preparation course":[self.test_preparation_course],
                "reading score":[self.reading_score],
                "writing score":[self.writing_score]


            }

            return pd.DataFrame(custom_data_input_dict)
                      
        except Exception as e:
            raise CustomException(e,sys)

        

        