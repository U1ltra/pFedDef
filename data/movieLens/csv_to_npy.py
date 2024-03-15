
import csv
import numpy as np

def csv_to_npy(csv_path, npy_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = np.array(data)
    np.save(npy_path, data)

# csv_to_npy('ml-25m/ratings.csv', 'ml-25m/ratings.npy')
csv_to_npy('ml-25m/movies.csv', 'ml-25m/labels.npy')
