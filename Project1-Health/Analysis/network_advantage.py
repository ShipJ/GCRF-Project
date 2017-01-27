''' Two Network Advantage metrics: 1. Median degree, 2. Normalised Entropy '''
# Input:
# Output:

import pandas as pd
import numpy as np
import sys

if __name__ == "__main__":

    # country = sys.argv[1]
    country = 'Senegal'

    path = "/Users/JackShipway/Desktop/UCLProject/Data/%s" % country

    # Set known data set values (number of towers, and time length of data)
    if country == 'Senegal':
        num_bts, hours = 1668, 8760
    elif country == 'IvoryCoast':
        num_bts, hours = 1240, 3360
    else:
        num_bts, hours = 10000, 100000

    total_activity = np.genfromtxt(path + "/CDR/Metrics/total_activity.csv", delimiter=',', skiprows=1)
    adj_matrix = np.genfromtxt(path+'/CDR/StaticMetrics/Other/adj_matrix.csv', delimiter=',')

    ''' Median Degree '''

    for i in range(1240):
        row_col = np.concatenate((adj_matrix[i, :], adj_matrix[:, i]))
        row_col_self = np.delete(row_col, i)
        median_weight = np.median(row_col_self) - 0.1
        adj_matrix[:, i][np.where(adj_matrix[i, :] < median_weight)] = 0
        adj_matrix[i, :][np.where(adj_matrix[:, i] < median_weight)] = 0

    adj_matrix[adj_matrix > 0] = 1

    total_deg_matrix = np.zeros(1240)
    in_deg_matrix = np.zeros(1240)
    out_deg_matrix = np.zeros(1240)
    for i in range(1240):
        total_deg_matrix[i] = np.sum(np.delete(np.concatenate((adj_matrix[i, :], adj_matrix[:, i])), i))
        in_deg_matrix[i] = np.sum(adj_matrix[i, :])
        out_deg_matrix[i] = np.sum(adj_matrix[:, i])



    ''' Normalised Entropy '''
    # q_log = np.genfromtxt("q_log.csv", delimiter=',')
    # where_nan = np.isnan(q_log)
    # q_log[where_nan] = 0
    # deg_vector = np.genfromtxt("deg_vector.csv", delimiter=',')
    #
    # entropy = np.zeros(1240)
    #
    # for i in range(1240):
    #     sum_row = np.sum(q_log[i, :], axis=0)
    #     entropy[i] = (-1*sum_row) / np.log10(deg_vector[i])
    #
    # np.savetxt("entropy.csv", entropy, delimiter=',')

    # entropy = np.genfromtxt("entropy.csv", delimiter=',')
    #
    # # Proportion of each cell tower associated with each administrative region
    # IntersectPop = pd.DataFrame(pd.read_csv("/Users/JackShipway/Desktop/UCLProject/Project1-Health/Data/Temporary/IvoryCoastIntersectPop.csv"))
    #
    # adm_1_pop = IntersectPop.groupby('Adm_1')['Pop_2010'].sum().reset_index()
    # adm_2_pop = IntersectPop.groupby('Adm_2')['Pop_2010'].sum().reset_index()
    # adm_3_pop = IntersectPop.groupby('Adm_3')['Pop_2010'].sum().reset_index()
    # adm_4_pop = IntersectPop.groupby('Adm_4')['Pop_2010'].sum().reset_index()
    #
    # cell_tower_adm = pd.DataFrame(pd.read_csv(path+"/Essential/CellTower_Adm_1234.csv"))
    # list = np.array(cell_tower_adm['CellTowerID'])
    # entropy = entropy[list]
    # cell_tower_adm['entropy'] = entropy
    #
    # cell_towers_adm_1 = cell_tower_adm.groupby('ID_1')['ID_1'].count().reset_index(drop=True)
    # cell_towers_adm_2 = cell_tower_adm.groupby('ID_2')['ID_2'].count().reset_index(drop=True)
    # cell_towers_adm_3 = cell_tower_adm.groupby('ID_3')['ID_3'].count().reset_index(drop=True)
    # cell_towers_adm_4 = cell_tower_adm.groupby('ID_4')['ID_4'].count().reset_index(drop=True)
    #
    # entropy_adm_1 = cell_tower_adm.groupby('ID_1')['entropy'].sum().reset_index()['entropy'] * (adm_1_pop['Pop_2010'] / cell_towers_adm_1)
    # entropy_adm_2 = cell_tower_adm.groupby('ID_2')['entropy'].sum().reset_index()['entropy'] * (adm_2_pop['Pop_2010'] / cell_towers_adm_2)
    # entropy_adm_3 = cell_tower_adm.groupby('ID_3')['entropy'].sum().reset_index()['entropy'] * (adm_3_pop['Pop_2010'] / cell_towers_adm_3)
    # entropy_adm_4 = cell_tower_adm.groupby('ID_4')['entropy'].sum().reset_index()['entropy'] * (adm_4_pop['Pop_2010'] / cell_towers_adm_4)

    # np.savetxt("entropy_adm_1.csv", entropy_adm_1, delimiter=',')
    # np.savetxt("entropy_adm_2.csv", entropy_adm_2, delimiter=',')
    # np.savetxt("entropy_adm_3.csv", entropy_adm_3, delimiter=',')
    # np.savetxt("entropy_adm_4.csv", entropy_adm_4, delimiter=',')

    IntersectPop = pd.DataFrame(
        pd.read_csv("/Users/JackShipway/Desktop/UCLProject/Project1-Health/Data/Temporary/IvoryCoastIntersectPop.csv"))

    adm_1_pop = IntersectPop.groupby('Adm_1')['Pop_2010'].sum().reset_index()
    adm_2_pop = IntersectPop.groupby('Adm_2')['Pop_2010'].sum().reset_index()
    adm_3_pop = IntersectPop.groupby('Adm_3')['Pop_2010'].sum().reset_index()
    adm_4_pop = IntersectPop.groupby('Adm_4')['Pop_2010'].sum().reset_index()

    cell_tower_adm = pd.DataFrame(pd.read_csv(path + "/Essential/CellTower_Adm_1234.csv"))


    ''' total degree '''
    total_deg_matrix = total_deg_matrix[list]
    cell_tower_adm['total_deg'] = total_deg_matrix

    cell_towers_adm_1 = cell_tower_adm.groupby('ID_1')['ID_1'].count().reset_index(drop=True)
    cell_towers_adm_2 = cell_tower_adm.groupby('ID_2')['ID_2'].count().reset_index(drop=True)
    cell_towers_adm_3 = cell_tower_adm.groupby('ID_3')['ID_3'].count().reset_index(drop=True)
    cell_towers_adm_4 = cell_tower_adm.groupby('ID_4')['ID_4'].count().reset_index(drop=True)

    total_deg_adm_1 = cell_tower_adm.groupby('ID_1')['total_deg'].sum().reset_index()['total_deg'] * (
    adm_1_pop['Pop_2010'] / cell_towers_adm_1)
    total_deg_adm_2 = cell_tower_adm.groupby('ID_2')['total_deg'].sum().reset_index()['total_deg'] * (
    adm_2_pop['Pop_2010'] / cell_towers_adm_2)
    total_deg_adm_3 = cell_tower_adm.groupby('ID_3')['total_deg'].sum().reset_index()['total_deg'] * (
    adm_3_pop['Pop_2010'] / cell_towers_adm_3)
    total_deg_adm_4 = cell_tower_adm.groupby('ID_4')['total_deg'].sum().reset_index()['total_deg'] * (
    adm_4_pop['Pop_2010'] / cell_towers_adm_4)

    np.savetxt("total_deg_adm_1.csv", total_deg_adm_1, delimiter=',')
    np.savetxt("total_deg_adm_2.csv", total_deg_adm_2, delimiter=',')
    np.savetxt("total_deg_adm_3.csv", total_deg_adm_3, delimiter=',')
    np.savetxt("total_deg_adm_4.csv", total_deg_adm_4, delimiter=',')
    
    ''' in-degree '''
    in_deg_matrix = in_deg_matrix[list]
    cell_tower_adm['in_deg'] = in_deg_matrix

    cell_towers_adm_1 = cell_tower_adm.groupby('ID_1')['ID_1'].count().reset_index(drop=True)
    cell_towers_adm_2 = cell_tower_adm.groupby('ID_2')['ID_2'].count().reset_index(drop=True)
    cell_towers_adm_3 = cell_tower_adm.groupby('ID_3')['ID_3'].count().reset_index(drop=True)
    cell_towers_adm_4 = cell_tower_adm.groupby('ID_4')['ID_4'].count().reset_index(drop=True)

    in_deg_adm_1 = cell_tower_adm.groupby('ID_1')['in_deg'].sum().reset_index()['in_deg'] * (
        adm_1_pop['Pop_2010'] / cell_towers_adm_1)
    in_deg_adm_2 = cell_tower_adm.groupby('ID_2')['in_deg'].sum().reset_index()['in_deg'] * (
        adm_2_pop['Pop_2010'] / cell_towers_adm_2)
    in_deg_adm_3 = cell_tower_adm.groupby('ID_3')['in_deg'].sum().reset_index()['in_deg'] * (
        adm_3_pop['Pop_2010'] / cell_towers_adm_3)
    in_deg_adm_4 = cell_tower_adm.groupby('ID_4')['in_deg'].sum().reset_index()['in_deg'] * (
        adm_4_pop['Pop_2010'] / cell_towers_adm_4)

    np.savetxt("in_deg_adm_1.csv", in_deg_adm_1, delimiter=',')
    np.savetxt("in_deg_adm_2.csv", in_deg_adm_2, delimiter=',')
    np.savetxt("in_deg_adm_3.csv", in_deg_adm_3, delimiter=',')
    np.savetxt("in_deg_adm_4.csv", in_deg_adm_4, delimiter=',')


    ''' out degree '''
    out_deg_matrix = out_deg_matrix[list]
    cell_tower_adm['out_deg'] = out_deg_matrix

    cell_towers_adm_1 = cell_tower_adm.groupby('ID_1')['ID_1'].count().reset_index(drop=True)
    cell_towers_adm_2 = cell_tower_adm.groupby('ID_2')['ID_2'].count().reset_index(drop=True)
    cell_towers_adm_3 = cell_tower_adm.groupby('ID_3')['ID_3'].count().reset_index(drop=True)
    cell_towers_adm_4 = cell_tower_adm.groupby('ID_4')['ID_4'].count().reset_index(drop=True)

    out_deg_adm_1 = cell_tower_adm.groupby('ID_1')['out_deg'].sum().reset_index()['out_deg'] * (
        adm_1_pop['Pop_2010'] / cell_towers_adm_1)
    out_deg_adm_2 = cell_tower_adm.groupby('ID_2')['out_deg'].sum().reset_index()['out_deg'] * (
        adm_2_pop['Pop_2010'] / cell_towers_adm_2)
    out_deg_adm_3 = cell_tower_adm.groupby('ID_3')['out_deg'].sum().reset_index()['out_deg'] * (
        adm_3_pop['Pop_2010'] / cell_towers_adm_3)
    out_deg_adm_4 = cell_tower_adm.groupby('ID_4')['out_deg'].sum().reset_index()['out_deg'] * (
        adm_4_pop['Pop_2010'] / cell_towers_adm_4)

    np.savetxt("out_deg_adm_1.csv", out_deg_adm_1, delimiter=',')
    np.savetxt("out_deg_adm_2.csv", out_deg_adm_2, delimiter=',')
    np.savetxt("out_deg_adm_3.csv", out_deg_adm_3, delimiter=',')
    np.savetxt("out_deg_adm_4.csv", out_deg_adm_4, delimiter=',')


    ''' Gravity Residual '''

    













