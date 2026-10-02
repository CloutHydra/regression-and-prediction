import numpy as np
from scipy.stats import linregress

inputs = np.array([1, 2, 3, 4])
outputs = np.array([2.1, 2.9, 3.7, 4.6])

data_dict = dict(zip(inputs, outputs))
slope, intercept, r_value, p_value, std_err = linregress(inputs, outputs)

user_input = float(input("Enter an input value: "))

if user_input in data_dict:
    corresponding_output = data_dict[user_input]
    print(f"Found corresponding output value: {corresponding_output}")
else:
    pred_value = slope * user_input + intercept
    print("Did not find user input value in the dictionary")
    print(f"Predicted Output: {pred_value}")
