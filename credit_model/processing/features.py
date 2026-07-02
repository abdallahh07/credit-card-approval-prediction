import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AgeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x = x.copy()
        x["AGE_YEARS"] = (-x["DAYS_BIRTH"] / 365).astype(int)
        x = x.drop(columns=["DAYS_BIRTH"])
        return x

class EmploymentTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x = x.copy()
        x["DAYS_EMPLOYED"] = x["DAYS_EMPLOYED"].replace(365243, np.nan)
        x["YEARS_EMPLOYED"] = (-x["DAYS_EMPLOYED"] / 365)
        x = x.drop(columns=["DAYS_EMPLOYED"])
        return x