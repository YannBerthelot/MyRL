import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

report_folder = "reports"
color_dict = {
    "Q-learning": "blue",
    "SARSA": "green",
    "Double-Q-learning": "purple",
    "DynaQ": "red",
}
num_dict = {
    "Q-learning": 0,
    "SARSA": 1,
    "Double-Q-learning": 2,
    "DynaQ": 3,
}


def plot_cumsum():
    fig, ax = plt.subplots(1, len(color_dict), figsize=(20, 10), sharey=True)
    for root, dirs, files in os.walk(report_folder):
        path = root.split(os.sep)
        for file in [file for file in files if file.endswith(".json")]:
            file_path = "/".join(path)
            with open(os.path.join(file_path, file)) as f:
                file_data = json.load(f)
                dict_file = eval(file[:-5])
                env = dict_file["env"]
                cum_reward = np.cumsum(
                    [np.sum(episode) for episode in file_data["training_rewards"]]
                )
                pd.Series(cum_reward).plot(
                    ax=ax[num_dict[dict_file["agent"]]],
                    label=f'{dict_file["agent"]}',
                    color=color_dict[dict_file["agent"]],
                )
    plt.legend(
        handles=[
            Line2D([0], [0], label=agent, color=color)
            for agent, color in color_dict.items()
        ]
    )
    plt.savefig("Cumulative rewards.png")


plot_cumsum()