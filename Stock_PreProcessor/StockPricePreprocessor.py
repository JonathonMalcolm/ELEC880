import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


G_directory = 'Data/'
G_threshold = 0.5 # in percent

j = 0
for filename in os.listdir(G_directory):
    plot_data = []
    # if j > 5:
    #     break
    # j += 1
    if filename.endswith(".csv"):
        cur_file = filename
        data = pd.read_csv(G_directory + cur_file)
        new_data = pd.DataFrame(data['Date'].copy())
        new_data['Open'] = data['Open'].pct_change()
        new_data['High'] = data['High'].pct_change()
        new_data['Low'] = data['Low'].pct_change()
        new_data['Close'] = data['Close'].pct_change()
        new_data['Adj Close'] = data['Adj Close'].diff()
        new_data['Volume'] = data['Volume']

        for thresh in np.arange(0.2, 4, 0.2):
            final_data = pd.DataFrame()

            row_data = []
            count = 0
            for index, row in new_data.iterrows():
                if abs(row['Close']) < thresh / 100:
                    row_data.append(0)
                    count += 1
                else:
                    row_data.append(1)

            final_data['Date'] = data['Date'].copy()
            final_data[cur_file] = row_data

            percent_lost = count / len(row_data)

            plot_data.append([thresh, percent_lost])

        df = pd.DataFrame(plot_data, columns=['K', 'Percent'])

        plt.plot(df['K'], df['Percent'], label=cur_file)

        continue
    else:
        continue
plt.xlabel('K Value')
plt.ylabel('Fraction Lost')
plt.legend(loc='best')
plt.show()