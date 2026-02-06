import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

#to read the csv files
def read_csv():
    teams= ["Atlanta_Hawks", "Boston_Celtics", "Brooklyn_Nets", "Charlotte_Hornets", "Chicago_Bulls", "Cleveland_Cavaliers", 
                 "Dallas_Mavericks", "Denver_Nuggets", "Detroit_Pistons", "Golden_State_Warriors", "Houston_Rockets", "Indiana_Pacers",
                 "Los_Angeles_Clippers", "Los_Angeles_Lakers", "Memphis_Grizzlies", "Miami_Heat", "Milwaukee_Bucks", "Minnesota_Timberwolves",
                 "New_Orleans_Pelicans", "New_York_Knicks", "Oklahoma_City_Thunder", "Orlando_Magic", "Philadelphia_76ers", "Phoenix_Suns",
                 "Portland_Trail_Blazers", "Sacramento_Kings", "San_Antonio_Spurs", "Toronto_Raptors", "Utah_Jazz", "Washington_Wizards"]
    
    #season = 2025-26
    
    folder = Path(__file__).parent.parent / "collected data" / "2025-26 season"
    all_teams = []

    for team in teams:
        df = pd.read_csv(folder / f"{team}_2025-26.csv")
        all_teams.append(df)
    
    return pd.concat(all_teams, ignore_index=True)
        
df = read_csv()
print(df)

