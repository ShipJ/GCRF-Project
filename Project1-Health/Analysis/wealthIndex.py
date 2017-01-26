import pandas as pd
import numpy as np

if __name__ == "__main__":

    wealth = pd.DataFrame(pd.read_csv("Data/IvoryCoast/DHS/Extracted/Wealth.csv", encoding="utf-8-sig"))

    quintile_poorest = np.zeros(9686)
    for i in range(len(wealth)):
        if wealth['WealthQuintile'].iloc[i] == 1:
            quintile_poorest[i] = 1
        else:
            quintile_poorest[i] = 0

    wealth['QuintilePoorest'] = quintile_poorest

    poverty_rate = pd.DataFrame(wealth.groupby(by='ClustID')['QuintilePoorest'].mean())


    # Aggregate at the shapefile level - 191 regions rather than 351
    corresponding_regions = pd.DataFrame(pd.read_csv("Data/IvoryCoast/DHS/ClustIDAdm_4.csv",
                                                     encoding="utf-8-sig",
                                                     usecols=['ClustID', 'Adm_4ID']))


    # Remove regions for which there is no data (10 in total)
    poverty_rate = poverty_rate.iloc[corresponding_regions['ClustID']-1]

    poverty_rate['Adm_4ID'] = np.array(corresponding_regions['Adm_4ID'])

    poverty_rate_adm_4 = poverty_rate.groupby(by='Adm_4ID')['QuintilePoorest'].mean()

    print poverty_rate_adm_4.to_csv('Data/IvoryCoast/DHS/Extracted/FinalWealthIndex.csv', index=True)








    # subnational_wealth = wealth.groupby(by='HV001')['HV271'].median()
    # subnational_wealth = np.array(subnational_wealth.ix[corresponding_regions['ClusterID']])
    # corresponding_regions['WealthMedian'] = subnational_wealth
    # corresponding_regions = corresponding_regions.dropna()
    #
    # admin_level_1 = corresponding_regions.groupby(by='SubnationalID')['WealthMedian'].mean()
    #
    # cell_tower_adm_1 = pd.DataFrame(pd.read_csv("Data/IvoryCoast/CDR/Metrics/activity_per_adm_1.csv",
    #                                                encoding="utf-8-sig", usecols=['ID', 'ID_1']))
    # cell_tower_activity = np.load("Data/IvoryCoast/CDR/Metrics/total_activity.npy")
    #
    # cell_tower_adm_1['activity'] = cell_tower_activity[[cell_tower_adm_1['ID'] - 1]]
    #
    # cell_tower_adm_1 = cell_tower_adm_1.loc[~(cell_tower_adm_1['activity'] == 0)]
    # cell_tower_adm_1 = cell_tower_adm_1.loc[1:]
    # a = pd.DataFrame(cell_tower_adm_1.groupby(by='ID_1')['activity'].mean())
    #
    # # a.to_csv('Data/IvoryCoast/CDR/Metrics/adm_4_activity.csv')
    #
    # pop_per_191 = pd.DataFrame(pd.read_csv('Data/IvoryCoast/DHS/pop_per_191regions.csv'))
    # pop_per_191 = pd.DataFrame(pop_per_191.drop([5, 16]))
    #
    # b = np.array(pop_per_191.groupby(by='ID_1')['sum'].sum())
    # a = np.array(a.drop([0]))
    # c = [a[i, 0] / float(b[i]) for i in range(14)]
    #
    # # print c
    #
    # from scipy.stats.stats import pearsonr
    #
    # print admin_level_1
    #
    # c = pearsonr(c, admin_level_1[1:])
    # print c
    #














    #
    # plt.plot(range(15), a, range(15), b)
    # plt.show()
    #
    #
    # c = pearsonr(a, b)
    # print c