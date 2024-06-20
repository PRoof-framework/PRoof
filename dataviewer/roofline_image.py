import matplotlib.pyplot as plt
import pandas as pd
import itertools
from adjustText import adjust_text

# plt.axes(yscale = "log", xscale = "log")


# Define the graph properties
# fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

# plt.suptitle('Roofline Plots')

# Initialize the plot counter
plt_cnt = 1

hw = 'GPU'
tpp_hw_ = 100000    # GFLOPS
mem_bw_hw_ = 1500   # GB/s
scale = 'T' # G or T


# Load the data for the current hardware from a CSV file
data = pd.read_csv(f'data.csv')

data_gflops = [data['model.bench.results.*.layer_prof.*.flops'][i] / data['model.bench.results.*.layer_prof.*.median_time'][i] / (1e9 if scale == 'G' else 1e12) for i in range(len(data['model.bench.results.*.layer_prof.*.flops']))]
tpp_hw = tpp_hw_
mem_bw_hw = mem_bw_hw_
if scale == 'T':
    tpp_hw /= 1000
    mem_bw_hw /= 1000

flops_byte = [data['model.bench.results.*.layer_prof.*.flops'][i] / data['model.bench.results.*.layer_prof.*.memory'][i] if data['model.bench.results.*.layer_prof.*.memory'][i] else 0 for i in range(len(data['model.bench.results.*.layer_prof.*.flops']))]

# Define the plot
# plt.axes(yscale = "log", xscale = "log")
plt.title('model')
plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
plt.ylabel(f'Performance ({scale}FLOPS)')

# Plot the memory boundaries
plt.axhline(y=tpp_hw, color='gray', linestyle='dashed')
# plt.text(0.2, tpp_hw+10, 'DRAM', ha='center', va='center', fontweight='bold')

color = []
for i, x in enumerate(data['model.bench.results.*.layer_prof.*.name']):
    if 'Conv' in x:
        color.append('red')
    elif 'MatMul' in x:
        color.append('green')
    elif data['model.bench.results.*.layer_prof.*.flops'][i] == 0:
        color.append('blue')
    else:
        color.append('yellow')



# Plot the data points
plt.scatter(flops_byte, data_gflops, color=color, s=50, alpha=[min(1, x / 20) for x in data['model.bench.results.*.layer_prof.*.time_percentage']])

# Add labels to the data points
# for i, name in enumerate(data['name']):
#     plt.annotate(i, (data['flops_byte'][i], data_gflops[i]), xytext=(3, 3), textcoords='offset points', fontsize=8, color='blue')
    # plt.annotate(name, (data['flops_byte'][i], data_gflops[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)

# texts = []
# for j, (x, y) in enumerate(zip(flops_byte, data_gflops)):
#     texts.append(axs[(plt_cnt-1)//2, (plt_cnt-1)%2].text(x, y, f"{j+1}", fontsize=8, color='blue', ha='center', va='center'))
# # adjust_text(texts)
# adjust_text(texts, force_text=(0.2, 0.2))

# Plot the performance and arithmetic intensity limits
# plt.plot(data['flops_byte'], [min(tpp_hw/i, mem_bw_hw) for i in data['flops_byte']], 'g--')

# Add the memory wall
slope = mem_bw_hw
x_vals = [0, tpp_hw/mem_bw_hw]
y_vals = [0, tpp_hw]
plt.plot(x_vals, y_vals, color='gray', linestyle='dashed')



# Adjust the padding between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.27, bottom=0.1, top=0.95, left=0.09, right=0.99)

# Save the figure
plt.savefig('roofline.png')
plt.savefig('roofline.pdf')
