from data_loader import DataLoader
print("running main.py")
etf_symbol = "XLE"
auxiliary_symbols=["CL=F", "EURUSD=X"]

DataLoader(etf_symbol, auxiliary_symbols=auxiliary_symbols)
