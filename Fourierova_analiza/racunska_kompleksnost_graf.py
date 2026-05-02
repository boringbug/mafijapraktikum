import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


file_dir = "/home/flopa/Physics/mafijapraktikum/Fourierova_analiza/time_complexity.csv"
complexity_data = pd.read_csv(file_dir)

elapsed_time = complexity_data['Sampling rate']
sampling_rate = complexity_data['Time(ns)']

plt.loglog(sampling_rate, elapsed_time/1e9)
plt.xlabel('N')
plt.ylabel('Time')
plt.show()
