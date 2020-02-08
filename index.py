import pandas as pd
import numpy as np
import pyplot as plt

import os.path
file = 'source/train.csv'
# load train data
isFileExist = os.path.exists(file);
if (isFileExist):
    data = pd.read_csv(file, header=1)
    #drop NULL value to avoid learning prob
    data = pd.DataFrame(data)
    data_no_null = data.dropna()
    print (data_no_null.count())


