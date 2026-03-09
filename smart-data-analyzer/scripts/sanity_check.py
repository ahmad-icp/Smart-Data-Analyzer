import pandas as pd
from modules.data_loader import load_dataset_bytes
from modules.ai_cleaning import analyze_dataset

csv = 'a,b\n1,2\n3,\n'
df = load_dataset_bytes(csv.encode('utf-8'), 'test.csv')
print('df:')
print(df)
print('suggestions:')
print(analyze_dataset(df))
