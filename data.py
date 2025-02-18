from ucimlrepo import fetch_ucirepo 
import pandas as pd

wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 

df = pd.DataFrame(X)
df['Quality'] = y

df.to_csv("data/dataset.csv", index=False)