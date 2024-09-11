# attention_to_glycemia

This is the code used in the paper "Forecasting Glucose Values for Patients with Type 1 Diabetes Using Heart Rate Data." The code takes as input a CSV file containing measured heart rate (HR) and blood glucose (BG) levels of patients with type 1 diabetes. It splits the data into overlapping time series windows of varying sizes, which are then used as input to a GRU (Gated Recurrent Unit) with an attention mechanism to forecast future blood glucose levels.

The algorithm divides the dataset into two arrays: one containing only HR values and the other containing only BG values. The GRU outputs for the two arrays are then concatenated, and the same is done with the attention mechanism. Due to the diversity of datasets used, the user can modify several parameters:

The split between the test and training sets.
In the prepare_sequences function, it is possible to adjust the prediction horizon. In the line y.append(data[i + sequence_length, 0]), if the dataset has values measured every 5 minutes, this line corresponds to a prediction horizon of 5 minutes. Changing it to y.append(data[i + sequence_length + 1, 0]) sets the prediction horizon to 10 minutes, and so on.
The sequence length processed by the algorithm to produce the output.
At the end of the code, there are lines that allow the user to calculate the prediction error.
