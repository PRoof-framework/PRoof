from io import StringIO
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FuncFormatter
import pandas as pd
import itertools
# from adjustText import adjust_text
import argparse

import read_value

parser = argparse.ArgumentParser(
    prog = "PRoof roofline image render",
    description = "",
    epilog = "WIP",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("report_file", help="e.g. report.json")
parser.add_argument("-b", "--batch-size", default='*')
parser.add_argument("-t", "--title", default='')
parser.add_argument("--log", action='store_true', help="log scale axes")
args = parser.parse_args()

json_file = args.report_file
report_name = Path(args.report_file).stem


# Define the graph properties
plt.subplots(nrows=2, ncols=2, figsize=(8, 5), gridspec_kw={'height_ratios': [4, 1], 'width_ratios': [7, 1]})

# plt.suptitle('Roofline Plots')

# Initialize the plot counter
plt_cnt = 1

hw = 'GPU'
tpp_hw_ = 4000     # GFLOPS
mem_bw_hw_ = 132.9     # GB/s
scale = 'T' # G or T

# Load the data
key_time_percentage = f'model.bench.results.{args.batch_size}.layer_prof.*.time_percentage'
key_flops = f'model.bench.results.{args.batch_size}.layer_prof.*.flops'
key_memory = f'model.bench.results.{args.batch_size}.layer_prof.*.memory'
key_median_time = f'model.bench.results.{args.batch_size}.layer_prof.*.median_time'
key_name = f'model.bench.results.{args.batch_size}.layer_prof.*.name'
key_extra = f'model.bench.results.{args.batch_size}.layer_prof.*.extra'
keys = f'{key_time_percentage}; {key_flops}; {key_memory}; {key_median_time}; {key_name}; {key_extra}'
csv_string = read_value.get_data([json_file], keys, True)

data = pd.read_csv(StringIO(csv_string))

data_gflops = [data[key_flops][i] / data[key_median_time][i] / (1e9 if scale == 'G' else 1e12) for i in range(len(data[key_flops]))]
_scale = 1000 if scale == 'T' else 1
tpp_hw = tpp_hw_ / _scale
mem_bw_hw = mem_bw_hw_ / _scale

flops_byte = [data[key_flops][i] / data[key_memory][i] if data[key_flops][i] else 0 for i in range(len(data[key_flops]))]



#### FIG MAIN
ax = plt.subplot(2, 2, 1)
if args.log:
    ax.set_xscale('log')
    ax.set_yscale('log')
plt.title(args.title or 'Layers of '+report_name)
plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
plt.ylabel(f'Performance ({scale}FLOPS)')


color = []
for i, (x, n) in enumerate(zip(data[key_name], data[key_extra])):
    # if 'Conv' in x:
    #     color.append('red')
    # elif 'MatMul' in x:
    #     color.append('green')
    # elif data[key_flops0][i] == 0:
    #     color.append('blue')
    # else:
    #     color.append('purple')

    # For shufflenet_v2_x1_0
    # if 'stage2' in x:
    #     if 'branch2.3/Conv' in x:     # dw
    #         color.append('red')
    #     elif 'branch2.7/Relu' in x:   # pw
    #         color.append('green')
    #     else:
    #         color.append('grey')
    # else:
    #     color.append('grey')

    # # For shufflenet_v2_x1_0_mod2
    # if 'stage2' in x and 'stage2.0' not in x:
    #     if 'branch2.3/Conv' in x:     # comb
    #         color.append('green')
    #     else:
    #         color.append('grey')
    # else:
    #     color.append('grey')

    # mod Transpose
    # For shufflenet_v2_x1_0
    # if 'CopyNode' in x:
    #     color.append('green')
    # elif 'branch1.0/Conv' in x or 'branch2.3/Conv' in x:     # dw
    #     color.append('orange')
    # elif 'Conv' in x:     # other
    #     color.append('red')
    # elif 'Transpose' in x:
    #     color.append('blue')
    # else:
    #     color.append('grey')

    # mod Transpose
    # For shufflenet_v2_x1_0-mod4
    # if 'CopyNode' in x:
    #     color.append('green')
    # elif 'branch_proj.2' in x or 'branch_main.3' in x:     # dw
    #     color.append('orange')
    # elif 'Conv' in x:     # other
    #     color.append('red')
    # elif 'Transpose' in x or 'ForeignNode' in x:
    #     color.append('blue')
    # else:
    #     color.append('grey')

    if 'MatMul' in n or 'Gemm' in n or 'conv_dw' in n:     # dw
        color.append('green')
    elif 'conv_dw' in n:  # other
        color.append('orange')
    elif 'Conv' in n:     # other
        color.append('red')
    elif 'CopyNode' in x or 'Transpose' in n or data[key_flops][i] == 0:
        color.append('blue')
    elif 'PWN' in x or 'ReduceMean' in n:
        color.append('purple')
    else:
        color.append('gray')

ALPHA_FACTOR = 20   # small value for deeper
ALPHA_FACTOR_DIST = 5   # for Latency distribution fig, small value for deeper
# ALPHA_FACTOR = 5   # small value for deeper

# Plot the data points
data_time_pct = data[key_time_percentage]
alpha = [min(1, x / ALPHA_FACTOR) for x in data_time_pct]
alpha_dist = [min(1, x / ALPHA_FACTOR_DIST) for x in data_time_pct]
ax.scatter(flops_byte, data_gflops, color=color, s=50, alpha=alpha)

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



def draw_roof(flops, membw, color='gray', linestyle='dashed', text_offset_y=0.1, mem_only=False):
    # memory
    x_vals = [0, flops/membw]
    y_vals = [0, flops]
    plt.plot(x_vals, y_vals, color=color, linestyle=linestyle)

    angle = 180 / math.pi * math.atan(membw)
    text_shift_factor = 0.1 if args.log else 0.5
    text_x = flops/membw * text_shift_factor
    text_y = flops * text_shift_factor + text_offset_y
    plt.annotate(f'{membw * _scale:.1f} GB/s', (text_x, text_y),
                fontsize=8, color=color,
                rotation=angle, rotation_mode='anchor',
                transform_rotates_text=True)

    if not mem_only:
        # compute
        plt.axhline(y=flops, color='gray', linestyle='dashed')
        plt.annotate(f'{flops:.1f} {scale}FLOPS', (flops/membw, flops * 1.05),
                    fontsize=8, color=color)

draw_roof(tpp_hw, mem_bw_hw)
# draw_roof(tpp_hw, 62.0312/_scale, color='orange', mem_only=True)
# draw_roof(tpp_hw, 15.1774/_scale, color='red', mem_only=True)
plt.grid()



#### FIG2
ax = plt.subplot(2, 2, 2)
if args.log:
    ax.set_yscale('log')
plt.xlabel('Latency dist.')

data_time_ms = [x * 1000 for x in data[key_median_time]]
zipped_sorted = sorted(zip(data_gflops, data_time_ms, color, alpha_dist))
data_gflops_sorted = [x[0] for x in zipped_sorted]
data_time_ms_sorted = [x[1] for x in zipped_sorted]
data_time_ms_sorted_accumulate = data_time_ms_sorted[:]
for i in range(1, len(data_time_ms_sorted_accumulate)):
    data_time_ms_sorted_accumulate[i] += data_time_ms_sorted_accumulate[i-1]
color_sorted = [x[2] for x in zipped_sorted]
alpha_sorted = [x[3] for x in zipped_sorted]

# plt.gca().invert_xaxis()
ax.xaxis.set_major_locator(MultipleLocator(data_time_ms_sorted_accumulate[-1] / 5))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{(pos-1)*20}%"))
plt.xticks(rotation=90)
plt.xlim(left=0, right=data_time_ms_sorted_accumulate[-1])
plt.grid(axis='x')
# plt.grid()
# plt.plot(data_time_ms_sorted_accumulate, data_gflops_sorted, color='black', linestyle='solid')
for i in range(len(data_time_ms_sorted_accumulate)):
    plt.plot(
        [data_time_ms_sorted_accumulate[i]-data_time_ms_sorted[i], data_time_ms_sorted_accumulate[i]],
        [data_gflops_sorted[i], data_gflops_sorted[i]],
        color='black', linewidth=1.5, linestyle='solid', zorder=101)
for i in range(len(data_time_ms_sorted_accumulate)-1):
    plt.plot(
        [data_time_ms_sorted_accumulate[i], data_time_ms_sorted_accumulate[i]],
        [data_gflops_sorted[i], data_gflops_sorted[i+1]],
        color='black', linewidth=1.5, linestyle='solid', zorder=101)
plt.scatter(data_time_ms_sorted_accumulate, data_gflops_sorted, color=color_sorted, s=20, alpha=alpha_sorted)
plt.fill_betweenx(data_gflops_sorted, [0] * len(data_time_ms_sorted_accumulate), data_time_ms_sorted_accumulate,
                 facecolor='gray', alpha=0.2)
plt.plot([0], [tpp_hw])



#### FIG3
ax = plt.subplot(2, 2, 3)
if args.log:
    ax.set_xscale('log')
plt.ylabel('Latency \n distribution')

data_time_ms = [x * 1000 for x in data[key_median_time]]
zipped_sorted = sorted(zip(flops_byte, data_time_ms, color, alpha_dist))
flops_byte_sorted = [x[0] for x in zipped_sorted]
data_time_ms_sorted = [x[1] for x in zipped_sorted]
data_time_ms_sorted_accumulate = data_time_ms_sorted[:]
for i in range(1, len(data_time_ms_sorted_accumulate)):
    data_time_ms_sorted_accumulate[i] += data_time_ms_sorted_accumulate[i-1]
color_sorted = [x[2] for x in zipped_sorted]
alpha_sorted = [x[3] for x in zipped_sorted]

# TODO: may double it

ax.yaxis.set_major_locator(MultipleLocator(data_time_ms_sorted_accumulate[-1] / 5))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{(pos-1)*20}%"))
plt.ylim(bottom=0, top=data_time_ms_sorted_accumulate[-1])
plt.grid(axis='y')
# plt.grid()

# bar
# for i in range(len(data_time_ms_sorted_accumulate)):
    # plt.bar(flops_byte_sorted[i], data_time_ms_sorted[i], width=5, bottom=data_time_ms_sorted_accumulate[i]-data_time_ms_sorted[i], color=color_sorted[i], linestyle='solid', edgecolor=(0,0,0,0.3), linewidth=0.5, zorder=200)
    # plt.bar(flops_byte_sorted[i], data_time_ms_sorted[i], width=5, bottom=data_time_ms_sorted_accumulate[i]-data_time_ms_sorted[i], color=color_sorted[i], zorder=200)
# for i in range(len(data_time_ms_sorted_accumulate)-1):
    # plt.plot([flops_byte_sorted[i], flops_byte_sorted[i+1]], [data_time_ms_sorted_accumulate[i], data_time_ms_sorted_accumulate[i]], linewidth=0.5, color=(0,0,0,0.5))

for i in range(len(data_time_ms_sorted_accumulate)):
    plt.plot(
        [flops_byte_sorted[i], flops_byte_sorted[i]],
        [data_time_ms_sorted_accumulate[i]-data_time_ms_sorted[i], data_time_ms_sorted_accumulate[i]],
        color='black', linewidth=1.5, linestyle='solid', zorder=101)
for i in range(len(data_time_ms_sorted_accumulate)-1):
    plt.plot(
        [flops_byte_sorted[i], flops_byte_sorted[i+1]],
        [data_time_ms_sorted_accumulate[i], data_time_ms_sorted_accumulate[i]],
        color='black', linewidth=1.5, linestyle='solid', zorder=101)
plt.scatter(flops_byte_sorted, data_time_ms_sorted_accumulate, color=color_sorted, s=20, alpha=alpha_sorted, zorder=100)
plt.fill_between(flops_byte_sorted, [0] * len(flops_byte_sorted), data_time_ms_sorted_accumulate,
                 facecolor='gray', alpha=0.2)
plt.subplot(2, 2, 4).axis('off')

# Adjust the padding between subplots
plt.subplots_adjust(hspace=0.30, wspace=0.15, bottom=0.1, top=0.95, left=0.11, right=0.99)


# Save the figure
plt.savefig(report_name+'.png', dpi=500)
plt.savefig(report_name+'.pdf')
