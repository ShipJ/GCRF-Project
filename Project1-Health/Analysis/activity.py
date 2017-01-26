# Jack Shipway, 3/11/16, UCL GCRF Project
#
# Total activity and duration for each cell tower (incoming + outgoing calls)
#
# Input: hourly-stamped csv files of CDR data
# Output: a 1240 x 1 array containing the total activity and duration of each cell tower (10 of these entries will
#     remain at 0 as the IDs do not correspond to actual towers. Some others will remain at 0 despite corresponding
#     to real towers; they are just never active. I try to discard as many of these as possible in the subsequent
#     processing stage. I also remove towers that become active after months of inactivity.

import pandas as pd
import numpy as np
import math

if __name__ == "__main__":

    m, w, d, h = 0, 0, 0, 0

    volume_total = np.zeros(1667)
    volume_in = np.zeros(1667)
    volume_out = np.zeros(1667)

    duration_total = np.zeros(1667)
    duration_in = np.zeros(1667)
    duration_out = np.zeros(1667)

    # Sum output for each cell tower, for each hour of data
    for hour in range(8760):
        print "Reading Hour: %s" % hour
        # Read each of the temporal data sets in turn, convert to multi-dimensional array
        data = pd.read_csv("Data/Senegal/CDR/Temporal/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv" % (m, w, d, h),
                           usecols=["source", "target", "volume", "duration"]).as_matrix()

        # Only  cell towers active within that time-step
        active_towers = np.array(np.unique(np.concatenate([data[:, 0], data[:, 1]])))

        for cell_tower in active_towers:

            active_data = data[data[:, 0] == cell_tower]

            in_vol = np.sum(active_data[:, 2])
            out_vol = np.sum(data[data[:, 1] == cell_tower][:, 2])

            in_dur = np.sum(active_data[:, 3])
            out_dur = np.sum(data[data[:, 1] == cell_tower][:, 3])

            if active_data.size == 0:
                self_vol = 0
                self_dur = 0
            else:
                self_active = active_data[active_data[:, 1] == cell_tower]
                if self_active.size != 0:
                    self_vol = self_active[0][2]
                    self_dur = self_active[0][3]
                else:
                    self_vol = 0
                    self_dur = 0

            volume_total[cell_tower] += (in_vol + out_vol) - self_vol
            volume_in[cell_tower] += in_vol
            volume_out[cell_tower] += out_vol

            duration_total[cell_tower] += (in_dur + out_dur) - self_dur
            duration_in[cell_tower] += in_dur
            duration_out[cell_tower] += out_dur

        # Increment CDR directory
        h = int(math.fmod(h + 1, 24))
        if h == 0:
            d = int(math.fmod(d + 1, 7))
            if d == 0:
                w = int(math.fmod(w + 1, 4))
                if w == 0:
                    m += 1

    total_activity = pd.DataFrame()
    total_activity['ID'] = np.array(range(1668))
    total_activity['Vol'] = volume_total
    total_activity['Vol_in'] = volume_in
    total_activity['Vol_out'] = volume_out
    total_activity['Dur'] = duration_total
    total_activity['Dur_in'] = duration_in
    total_activity['Dur_out'] = duration_out

    total_activity.to_csv('Data/Senegal/CDR/Metrics/total_activity.csv', index=None)
