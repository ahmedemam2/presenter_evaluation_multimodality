from numpy import mean
from numpy import std
import csv
from matplotlib import pyplot
import pandas as pd
# data1 =[0.6510496736,	0.6274017692,	0.6464243531	,0.598919332	,0.6497219786	,0.6173507994	,0.650283277,	0.6186849475	,0.6464745998	,0.6228793859	,0.001312578246,	0.008038632304	,0.004625320435	,0.02848243713,	0.001020401716	,0.0102827549	,0.4874128604	,-0.1187833951]
# data2 =[0.6196882129	,0.5137556195	,0.5844220519	,0.4656979442,	0.6000614416	,0.4879392434,	0.5928007364,	0.477901876, 	0.6057382822,	0.5137556195	,0.0127101127,	0.01553635213	,0.03526616096	,0.04805767536,	0.02375301719	,0.02861429751,	-1.686596681,	-1.486498864]

#
# # open the csv file
#
# with open('MasscomDatasettemp.csv', 'r') as f:
#     # create reader object
#     reader = csv.reader(f)
#     # read the specific rows, excluding the specified column
#     data1 = [row[:0] + row[17+1:] for idx, row in enumerate(reader) if idx in [1, 3, 5]]
#
# with open('MasscomDatasettemp.csv', 'r') as f:
#     # create reader object
#     reader = csv.reader(f)
#     # read the specific rows, excluding the specified column
#     data2 = [row[:0] + row[17+1:] for idx, row in enumerate(reader) if idx in [1, 3, 5]]
# do something with the rows
# summarize

# import openpyxl
#
# # Load the workbook
# wb = openpyxl.load_workbook('MasscomDataset.xlsx')
# # max-xS	max-yS	min-xS	min-yS	mean-xS	mean-yS	median-xS	median-yS	mode-xS	mode-yS
# # stdv-xS	stdv-yS	peaktopeak-xS	peaktopeak-yS	interquart-xS	interquart-yS	kurtosis-xS	kurtosis-yS
# # Get the active sheet
# sheet = wb.active

# Get the data from the sheet
# data = sheet.values
import pandas as pd
data_frame = pd.read_excel('MasscomDataset.xlsx')

start_row = 0
end_row = 55

data = data_frame.iloc[start_row:end_row]
data1 = list(data)
data_frame = pd.read_excel('MasscomDataset.xlsx')

start_row = 57
end_row = 116

data = data_frame.iloc[start_row:end_row]
data2 = list(data)
# Create the visualization
import matplotlib.pyplot as plt

plt.plot(data)
plt.show()
# print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
# print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# plot
pyplot.scatter(data1,data2)
# pyplot.legend()
pyplot.show()