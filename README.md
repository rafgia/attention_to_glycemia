# AttentionToGlycemia: a tool for analyzing type 1 diabetes glucose data

This repository contains the source code and an example dataset from the scientific paper [1]. May you find it useful, please cite us and contact us for any enquiry.

The code allows the user to predict the blood glucose values at different prediction time windows, using both Blood Glucose and Heart Rate values as input. The algorithm uses a GRU (gated Recurrent Unit) layer with a Double Attention mechanism.

# How to use it 

*Input*: a CSV (Comma Separated Values) file containing heart rate (HR) and blood glucose (BG) levels measured from patients with type 1 diabetes (use the *Example_dataset.csv* dataset file); 
*Run*: load the source code from *model.py* in a python environment (for instance a Jupyter Notebook, Visual Studio Code, Google Colab...). 
*Parameters*: several parameters can be defined, such as (i) the split between the test and training sets and (ii) the sequence length processed by the algorithm to produce the output. 
  -  For (i), adjust the prediction horizon appropriately by modifying the *prepare_sequences* function (line "y.append(data[i + sequence_length, 0]", e.g. if the dataset has values measured every 5 minutes, this line corresponds to a prediction horizon of 5 minutes; whereas "y.append(data[i + sequence_length + 1, 0]" sets it to 10 minutes).
  -  For (ii): modify the length of sequence at line: "seq_lenght = 12"
  -  At the end of *model.py*, there are lines that allow the user to calculate the prediction error. 

*Note*: for the experiments we used Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R-squared and Mean Absolute Percentage Error (MAPE).

# References

[1] Giancotti R., Bosoni P., Vizza P., Tradigo G., Gnasso A., Guzzi P. H., Bellazzi R., Irace C., Veltri P., Forecasting Glucose Values for Patients with Type 1 Diabetes Using Heart Rate Data, Computer Methods and Programs in Biomedicine, 2024
