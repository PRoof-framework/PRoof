import math
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from adjustText import adjust_text

import argparse

parser = argparse.ArgumentParser(
    prog = "image render",
    description = "",
    epilog = "WIP",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("workdir", help="")
parser.add_argument("output_file_stem", help="")
args = parser.parse_args()

# plt.axes(yscale = "log", xscale = "log")


# Define the graph properties
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))
# plt.suptitle('Roofline Plots')

key_prefix = 'model.bench.results.*.layer_prof.*.'

models = ['(a) ResNet-50 fp16', '(b) ViT tiny fp16*', '(c) EfficientNet B4 fp16', '(d) EfficientNetV2-T fp16'] #, 'shufflenet_v2_x0_5']

tpp_hw_ = 312000
mem_bw_hw_ = 1555
# tpp_hw_ = 304763
# mem_bw_hw_ = 1365
scale = 'T'

plt_cnt = 1

# Iterate over the different hardwares
for model in models:

    # Load the data for the current hardware from a CSV file
    data = pd.read_csv(f'{args.workdir}/fig3_{plt_cnt}.csv')

    data_gflops = [data[key_prefix+'flops'][i] / data[key_prefix+'median_time'][i] / (1e9 if scale == 'G' else 1e12) for i in range(len(data[key_prefix+'flops']))]
    tpp_hw = tpp_hw_
    mem_bw_hw = mem_bw_hw_
    if scale == 'T':
        tpp_hw /= 1000
        mem_bw_hw /= 1000

    flops_byte = [data[key_prefix+'flops'][i] / data[key_prefix+'memory'][i] if data[key_prefix+'memory'][i] else 0 for i in range(len(data[key_prefix+'flops']))]

    # Define the subplot
    plt.subplot(2, 2, plt_cnt)
    plt.title(model)
    plt.xlabel('Arithmetic Intensity (FLOP/Byte)')
    plt.ylabel(f'Performance ({scale}FLOP/s)')

    # Plot the memory boundaries
    plt.axhline(y=tpp_hw, color='gray', linestyle='dashed')
    # plt.text(0.2, tpp_hw+10, 'DRAM', ha='center', va='center', fontweight='bold')

    color = 'red'
    if plt_cnt == 2:
        color = []
        for i, x in enumerate(data[key_prefix+'name']):
            if data[key_prefix+'flops'][i] == 0:
                color.append('blue')
            elif 'MatMul' in x:
                color.append('green')
            else:
                color.append('gray')

    if plt_cnt == 3 or plt_cnt == 4:
        color = []
        for x in data[key_prefix+'name']:
            if '/conv_pw/' in x:
                color.append('green')
            elif '/conv_dw/' in x:
                color.append('blue')
            # elif '/conv_pwl/' in x:
            #     color.append('yellow')
            elif 'conv' in x:
                color.append('red')
            else:
                color.append('gray')


    # Plot the data points
    plt.scatter(flops_byte, data_gflops, color=color, s=50, alpha=[min(1, x / 15) for x in data[key_prefix+'time_percentage']])

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

    _scale = 1000 if scale == 'T' else 1
    angle = 180 / math.pi * math.atan(mem_bw_hw)
    text_shift_factor = 0.5
    text_offset_y = tpp_hw * 0.05
    text_x = tpp_hw/mem_bw_hw * text_shift_factor
    text_y = tpp_hw * text_shift_factor + text_offset_y
    plt.annotate(f'{mem_bw_hw * _scale:.1f} GB/s', (text_x, text_y),
                fontsize=8, color='#222',
                rotation=angle, rotation_mode='anchor',
                transform_rotates_text=True)

    plt.annotate(f'{tpp_hw:.1f} {scale}FLOP/s', (max(flops_byte) * 0.7, tpp_hw * 0.92),
            fontsize=8, color='#222')

    # Increment the plot counter
    plt_cnt += 1



# Adjust the padding between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.27, bottom=0.1, top=0.95, left=0.09, right=0.99)

# Save the figure
plt.savefig(f'{args.output_file_stem}.png', dpi=240)
plt.savefig(f'{args.output_file_stem}.pdf')
