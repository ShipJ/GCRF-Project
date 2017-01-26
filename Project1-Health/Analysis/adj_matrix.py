import numpy as np
import math
import os.path

if __name__ == "__main__":

    path = 'Data/IvoryCoast/CDR/Temporal'
    m, w, d, h = 0, 0, 0, 0
    adj_matrix = np.zeros((1240, 1240))

    # Cycle through all files, update adjacency matrix
    for hour in range(3360):
        file = path+'/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv'%(m, w, d, h)
        if os.path.isfile(file):
            print "Reading Data Set: %s" % hour, '--- ' 'm:', m, 'w:', w, 'd:', d, 'h:', h
            cdr = np.genfromtxt(file,
                                delimiter=',', usecols=(1, 2, 3), skiprows=1)
            if cdr.size > 3:
                active_towers = np.array(np.unique(np.concatenate([cdr[:, 0], cdr[:, 1]])))
                for cell_tower in active_towers:
                    active_data = cdr[cdr[:, 0] == cell_tower]
                    for i in range(len(active_data)):
                        adj_matrix[cell_tower, active_data[i, 1]] += active_data[i, 2]
            # Increment CDR directory
            h = int(math.fmod(h + 1, 24))
            if h == 0:
                d = int(math.fmod(d + 1, 7))
                if d == 0:
                    w = int(math.fmod(w + 1, 4))
                    if w == 0:
                        m += 1
    np.savetxt('adj_matrix.csv', adj_matrix, delimiter=',')

