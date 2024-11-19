import h5py
import numpy as np
def readfile_input(list_title,path_2,type_name):
    num_list = 0
    for i in list_title:
        dirs = path_2 + i
        h5f_file = h5py.File(dirs, 'r')
        data_final = np.array(h5f_file[type_name])

        if num_list == 0:
            data_total = data_final
        else:
            data_total = np.concatenate((data_total, data_final), axis=1)
        num_list = num_list + 1
        # print(i)
        # print('data_shape',data_final.shape)
        print('data_total_shape',data_total.shape)


    return data_total

def readfile_output(list_title,path_2,type_name):
    num_list = 0
    for i in list_title:
        dirs = path_2 + i
        h5f_file = h5py.File(dirs, 'r')
        data_final = np.array(h5f_file[type_name])
        # print('data000',data_x0.shape)
        if num_list == 0:
            data_total = data_final
        else:
            data_total = np.concatenate((data_total, data_final), axis=0)
        num_list = num_list + 1
    return data_total