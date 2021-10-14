from data_loader import DataLoader
print("running main.py")
etf_symbol = "XLE"

DataLoader(etf_symbol, auxiliary_symbols=["CL=F", "EURUSD=X"])
