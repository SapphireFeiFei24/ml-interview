import numpy
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # csv file -> numpy matrix
    matrix = np.loadtxt("data.csv", delimiter=",")

    # handles missing values
    matrix = np.genfromtxt("data.csv", delimiter=",", missing_values="NA", filling_values=0)

    # mixed types
    df = pd.read_csv("data.csv")
    matrix = df.values
