import pandas as pd
import numpy as np
import math
import os.path

if __name__ == "__main__":

    path = '/Users/JackShipway/Desktop/UCLProject/Project1-Health/Data/Senegal/DHS/Extracted'
    m, w, d, h = 0, 0, 0, 0

    malaria_dhs = pd.DataFrame(pd.read_csv(path+'/malaria.csv'))
    malaria_dhs = malaria_dhs.applymap(lambda x: 0 if isinstance(x, basestring) and x.isspace() else x)
    data = malaria_dhs.as_matrix().astype(int)

    data = np.delete(data, np.where(data[:, 4:40] == 6)[0], axis=0)
    data = np.delete(data, np.where(data[:, 4:40] == 7)[0], axis=0)
    data = np.delete(data, np.where(data[:, 4:40] == 8)[0], axis=0)
    data = np.delete(data, np.where(data[:, 4:40] == 9)[0], axis=0)

    total = np.sum(data[:, 4:], axis=1)

    malaria_cases = pd.DataFrame(data[:, [0, 1, 3]], columns=['ClustID', 'HouseID', 'Members'])
    malaria_cases['Cases'] = pd.Series(total)
    cluster_cases = pd.DataFrame(malaria_cases.groupby('ClustID')['Members', 'Cases'].sum()).reset_index()
    cluster_cases['CasePerPerson'] = cluster_cases['Cases'] / cluster_cases['Members']

    print cluster_cases

    print malaria_cases[malaria_cases['Cases'] > 0]