# from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
# from stable_baselines3.common import results_plotter

# log_dir = "tmp/"
# plot_results([log_dir], num_timesteps=100, x_axis=results_plotter.X_TIMESTEPS, task_name="Training Curve")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file into a DataFrame without a header and name the column as 'r'
df = pd.read_csv('tmp/monitor.csv', names=['r'], index_col=False)
data = df.to_numpy()[2:, 0].astype(np.float32).reshape((-1, ))
# Extract data from the 3rd line (index 2) and the 'r' column

alpha = 0.05
filtered_data = np.zeros_like(data)
filtered_data[0] = data[0]

for i in range(1, filtered_data.shape[0]):
    filtered_data[i] =  alpha * data[i] + (1 - alpha) * filtered_data[i-1]

sample = data[-10000:].copy()
sample[sample <= 15] = 0
print(sample.shape)
print(sample.astype(np.bool_).sum() / 10000) 

# Plot the extracted data
plt.plot(data[:], linewidth=0.5)
plt.plot(filtered_data[:])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Plot of Data from CSV')
plt.grid()
plt.show()