import sys
import pandas as pd
from src.exception import CustomExecption
from src.utils import load_object



class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path='artifact\model.pkl'
            preproccessor_path='artifact\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprossessor=load_object(file_path)
            daat_scaled=preprossessor.transform
            preds=model.predict(daat_scaled)
                                                          
            return pred
        except Exception as e:
            raise CustomExecption(e,sys)                                                                                                                                                                                                                                                                                                                                                                                                                                              
        
        
        
        
class CustomData:
    def __init__(self,
                 gender:str,
                 race/ethnicity:str,
                 parental level of education,
                 lunch:str,
                 test preparation course:str,
                 readimg score:int,
                 writing score:int):
        self.gender=gender
        self.race/ethnicity=race/ethnicity
        self.parental level of education=parental level of education
        self.lunch=lunch
        self.test preparation course=test preparation course
        self.readimg score=readimg score
        self.writing score=writing score
        
    def get_data_as_data_frame(self):
        try:
        custom_data_input_dict = {
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score],
        }
        return pd.DataFrame(custom_data_input_dict)
        