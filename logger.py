# Description: This file contains the Logger class which is used to log messages to a file.
import pandas as pd
import matplotlib.pyplot as plt
class Logger:
    def __init__(self, log_file, header):
        self.log_file = log_file
        self.data_frame = pd.DataFrame(columns=header)

    def log(self, message: dict):
        # Check if the message has all the required keys
        for key in self.data_frame.columns:
            if key not in message:
                raise ValueError(f"Message is missing key: {key}")
        # Convert the message to a DataFrame and filter out empty or all-NA entries
        message_df = pd.DataFrame([message]).dropna(how='all')
        if not message_df.empty:
            self.data_frame = pd.concat([self.data_frame, message_df], ignore_index=True)

    def save(self):
        self.data_frame.to_csv(self.log_file, index=False)

    def plot_data(self):
        self.data_frame.plot()
        plt.show()