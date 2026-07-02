import pandas as pd
from pathlib import Path
from credit_model.config.config import config

def load_data() -> pd.DataFrame:
    data_path = Path(__file__).parent.parent.parent / config.data_folder
    
    app = pd.read_csv(data_path / config.train_data_file)
    credit = pd.read_csv(data_path / config.credit_record_file)
    
    credit["target"] = credit["STATUS"].isin(["2","3","4","5"]).astype(int)
    target = credit.groupby("ID")["target"].max().reset_index()
    
    df = app.merge(target, on="ID", how="inner")
    
    return df