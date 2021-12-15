def read_data(path): 
 path1 = open(path) 
 dataset = pd.read_csv(path1, index_col='YMD') 
 # data = dataset['T0000'][:30] 
 # print(dataset.head(5)) 
 return array(dataset) 
