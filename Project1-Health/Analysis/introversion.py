import pandas as pd
import numpy as np
import math
import sys

if __name__ == "__main__":
    # Set variables for each country
    ic, ic_towers, ic_hours = 'IvoryCoast', 1238, 3360

    total_activity = pd.DataFrame(pd.read_csv('Data/%s/CDR/Metrics/total_activity_adm_1.csv' % ic))
    print total_activity

    # introversion = []
    # for cell_tower in range(ic_towers):
    #     in_degree = np.where(total_activity[cell_tower, :] > 0)
    #     out_degree = np.where(total_activity[:, cell_tower] > 0)
    #     total_degree = np.unique(np.concatenate([in_degree[0], out_degree[0]]))
    #
    #     if total_degree.size > 1:
    #         self_degree = total_degree[0]
    #         else_degree = sum(total_degree[1:])
    #         introversion.append(self_degree/float(else_degree))
    #     else:
    #         introversion.append(0)
    #
    #
    #
    # activity = np.load("Data/IvoryCoast/CDR/Metrics/activity.npy")
    #
    # corresponding_sub_pref = pd.DataFrame(pd.read_csv("Data/IvoryCoast/CorrespondingSubPref.csv"))
    # corresponding_sub_pref['Activity'] = pd.Series(activity)
    #
    # print pd.groupby(corresponding_sub_pref[['TargetID', 'Activity']], by='TargetID').sum()

