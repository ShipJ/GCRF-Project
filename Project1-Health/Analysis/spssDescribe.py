import pandas as pd
import numpy as np

if __name__ == "__main__":

    hiv_summary = pd.DataFrame(pd.read_csv('Data/Temporary/output.csv'))
    cluster_in_adm_4 = pd.DataFrame(pd.read_csv('Data/IvoryCoast/DHS/SubPrefADM_4.csv'))
    cluster_in_adm_4 = cluster_in_adm_4[cluster_in_adm_4['ID_4'] != 0]

    a = np.setdiff1d(range(1, 353), hiv_summary['clustID'])
    b = np.setdiff1d(range(1, 353), cluster_in_adm_4['field_1'])
    c = np.setdiff1d(b, a)

    for val in c:
        hiv_summary = hiv_summary[hiv_summary['clustID'] != val]

    for val2 in [209, 225, 324]:
        cluster_in_adm_4 = cluster_in_adm_4[cluster_in_adm_4['field_1'] != val2]

    hiv_summary['ID_4'] = np.array(cluster_in_adm_4['ID_4'])
    hiv_qgis = hiv_summary.groupby(by='ID_4')['Sum', 'Mean'].mean()
    hiv_qgis = pd.DataFrame(hiv_qgis.reindex(range(1, 192), fill_value=0))
    hiv_qgis.to_csv('Data/IvoryCoast/DHS/Extracted/hivForQGIS.csv')




    #
    # print hiv_summary
    # print len(cluster_in_adm_4['ID_4'])


    # print len(hiv_summary)
    # print len(cluster_in_adm_4['ID_4'])
    # hiv_summary['ID_4'] = np.array(cluster_in_adm_4['ID_4'])
    #
    #
    # print hiv_summary[hiv_summary['ID_4'] == 0]


    # hiv_qgis = hiv_summary.groupby(by='ID_4', as_index=False)['Sum', 'Mean'].mean()
    # print hiv_qgis




    # hiv_summary.to_csv('Data/IvoryCoast/DHS/hivSummaryQgis.csv', index=None)