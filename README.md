# AttentionToGlycemia

This is the code used in the paper "Forecasting Glucose Values for Patients with Type 1 Diabetes Using Heart Rate Data." 

How to use:
Input: the code takes as input a CSV file containing measured heart rate (HR) and blood glucose (BG) levels of patients with type 1 diabetes (use the Example_dataset.csv);
Run model.py in a python environment (for instance a Jupyter Notebook, Visual Studio Code, Google Colab...)
User can define parameters: (i) the split between the test and training sets and (ii) the sequence length processed by the algorithm to produce the output.
For (i): in the prepare_sequences function, it is possible to adjust the prediction horizon. Set prediction horizon in the line y.append(data[i + sequence_length, 0]), if the dataset has values measured every 5 minutes, this line corresponds to a prediction horizon of 5 minutes. Changing it to y.append(data[i + sequence_length + 1, 0]) sets the prediction horizon to 10 minutes, and so on.
For (ii): to modify the lenght of sequence it needs to change the number of values at line: (seq_lenght = 12)

At the end of the code, there are lines that allow the user to calculate the prediction error. For the experiments we used Root Mean Sqaured Error (RMSE), Mean Absolut Error (MAE), R-squared and Mean Absolut Percentage Error (MAPE).
