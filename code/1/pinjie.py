import numpy as np

def data_pinjie(data):
    [a,b,c,d]=data.shape   # 16 , 200, 8000, 31
    for i in range(d):
        if i == 0:
            data_out = data[:,:,:,i]
        else:
            data_out =  np.concatenate((data_out, data[:,:,:,i]), axis=2)
    return data_out