import pandas as pd
from data_loader import DataLoader
print("running train.py")
pd.options.display.width = 0
company_code = "XLE"

#DataGenerator(company_code)
DataLoader(company_code)

# exit here, training is done with stock_keras.ipynb.