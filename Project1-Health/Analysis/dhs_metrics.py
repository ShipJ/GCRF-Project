import pandas as pd
import numpy as np
import sys
from scipy.stats.stats import pearsonr

if __name__ == "__main__":

    # Get wealth index per cluster
    wealth_index = pd.DataFrame(pd.read_csv("Data/IvoryCoast/DHS/WealthIndex.csv", encoding="utf-8-sig"))
    cluster_wealth = wealth_index.groupby(by=['ClustNum'])['Score'].median()

    # Get total activity per cell tower
    cell_activity = pd.DataFrame(np.load("Data/IvoryCoast/CDR/Metrics/activity.npy"), columns=['Volume'])

    # Get regions corresponding to each cell tower
    corresponding_regions = pd.DataFrame(pd.read_csv("Data/CorrespondingSubPref.csv", usecols=['InputID', 'TargetID']))

    # Population data per cell tower
    pop_data = pd.DataFrame(pd.read_csv("Data/CorrespondingSubPref.csv"))


    activity = cell_activity['Volume']
    population = pop_data['Population']

    for i in range(1238):
        pop = population[i]
        if pop > 0:
            activity[i] /= pop

    corresponding_regions['ActivityPP'] = pd.Series(activity, index=corresponding_regions.index)

    cluster_activity = np.zeros(351)
    for i in range(351):
        region_i = np.array(corresponding_regions['InputID'][corresponding_regions['TargetID'] == i+1])
        for j in range(len(region_i)):
            cluster_activity[i] += corresponding_regions['ActivityPP'][region_i[j]]

    # pop_data = pd.DataFrame(pd.read_csv("Data/voronoipop4.csv"))
    # print np.setdiff1d(range(1238), pop_data['ID'])

    # for subpref in sorted(pd.unique(pop_data['TargetID'])):
    #     cluster_activity[subpref] /= cluster_pop[subpref]
    #
    #
    a = (cluster_wealth + abs(min(cluster_wealth)))
    cluster_wealth = a / max(a)

    print pearsonr(cluster_activity, cluster_wealth)
