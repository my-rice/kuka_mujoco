# Description: This file contains the Logger class which is used to log messages to a file.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
class Logger:
    def __init__(self, log_file, header):
        self.log_file = log_file
        self.data_frame = pd.DataFrame(columns=header)

    def log(self, message: dict):
        # Check if the message has all the required keys
        for key in self.data_frame.columns:
            if key not in message:
                raise ValueError(f"Message is missing key: {key}")
        
        # Append the message to the data frame
        self.new_data = pd.DataFrame([message])
        self.data_frame = pd.concat([self.data_frame, self.new_data], ignore_index=True)

    def save(self):
        self.data_frame.to_csv(self.log_file, index=False)

    def plot_columns(self, output_file, columns_names=None, references=None):

        if columns_names is None:
            columns_names = self.data_frame.columns

        fig, axs = plt.subplots(len(columns_names), 1, figsize=(10, 5 * len(columns_names)))
        axs = np.atleast_1d(axs)

        
        time = self.data_frame['time']

        
        for i, column in enumerate(columns_names):
            if column == 'time':
                continue
            axs[i].plot(time, self.data_frame[column], label=column)
            if references is not None:
                axs[i].plot(time, references[i] * np.ones_like(time), label=f'{column}_ref', linestyle='--')
                            
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel(column)
            axs[i].legend()
        
        plt.tight_layout()
        plt.savefig(output_file)


    def plot_column(self, column_name, output_file):
        fig, ax = plt.subplots()
        ax.plot(self.data_frame['time'], self.data_frame[column_name])
        ax.set_xlabel('Time')
        ax.set_ylabel(column_name)
        plt.savefig(output_file)
        plt.close()
