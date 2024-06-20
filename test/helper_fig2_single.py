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

parser.add_argument("device_id", help="1 to 12")
parser.add_argument("csv_file", help="")
parser.add_argument("output_file_stem", help="")
args = parser.parse_args()


plt.axes(yscale = "log", xscale = "log")


# Define the graph properties
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 2.25))


### Theoretical performance values from hardware vendor, to draw the "roofs" in roofline chart

# Peak performances GFLOP/s of the different hardwares
tpp = [312000, 624000, 330000, 660000, 25000, 50000, 6000, 21000, 5555.2, 24, 5734.4, 11468.8]
# Memory bandwidth (in GB/s) of the different hardwares
mem_bw = [1555, 1555, 1008, 1008, 102.4, 102.4, 59.7, 59.7, 183.3125, 12.8, 119.472, 119.472]

###


# Initialize the plot counter
plt_cnt = int(args.device_id)

# Define the hardware types
hw_types = ['NVIDIA A100, fp16 bs=128', 'NVIDIA A100, int8 bs=128', 'NVIDIA RTX 4090, fp16 bs=128', 'NVIDIA RTX 4090, int8 bs=128', 'NVIDIA Jetson Orin NX, fp16 bs=32', 'NVIDIA Jetson Orin NX, int8 bs=32', 'NVIDIA Jetson Xavier NX, fp16 bs=32', 'NVIDIA Jetson Xavier NX, int8 bs=32', 'Intel Xeon Gold 6330, fp32 bs=16', 'Raspberry Pi 4B, fp32 bs=4', 'NPU of Intel Core Ultra9 185H, fp16 bs=1', 'NPU of Intel Core Ultra9 185H, int8 bs=1']
scales = 'TTTTTTTTTGTT'

colors = ['#f41b43', '#26870e', '#7db8d6', '#ff7f50', '#00543d', '#d58500', '#4c4b4d', '#d8b76d', '#00dd00', '#5f9ea0', '#d5a6bd', '#4b0082', '#8a2be2', '#937393', '#1e90ff', '#d61eef', '#70302c', '#e6beff', '#ffcc00', '#b3b3b3']

models = {
    'distilbert-base-uncased-finetuned-sst-2-english.onnx': 1,
    'sd14-unet.onnx': 2,
    'efficientnet_b0.onnx': 3,
    'efficientnet_b4.onnx': 4,
    'efficientnetv2_rw_t.onnx': 5,
    'efficientnetv2_rw_s.onnx': 6,
    'mixer_b16_224.onnx': 7,
    'mobilenetv2_050.onnx': 8,
    'mobilenetv2_100.onnx': 9,
    'resnet34.onnx': 10,
    'resnet50.onnx': 11,
    'shufflenet_v2_x0_5.onnx': 12,
    'shufflenet_v2_x1_0.onnx': 13,
    'shufflenet_v2_x1_0-mod.onnx': 14,
    'swin_tiny_patch4_window7_224.onnx': 15,
    'swin_small_patch4_window7_224.onnx': 16,
    'swin_base_patch4_window7_224.onnx': 17,
    'vit_tiny_patch16_224.onnx': 18,
    'vit_small_patch16_224.onnx': 19,
    'vit_base_patch16_224.onnx': 20,
}

_kp = 'model.bench.results.*.'

# Iterate over the different hardwares
hw, tpp_hw, mem_bw_hw, scale = hw_types[plt_cnt-1], tpp[plt_cnt-1], mem_bw[plt_cnt-1], scales[plt_cnt-1]
for _ in range(1):
    print("fig", hw)
    # Load the data for the current hardware from a CSV file
    data = pd.read_csv(args.csv_file)

    data_gflops = [x / (1e9 if scale == 'G' else 1e12) for x in data[_kp+'flops_avg']]
    if scale == 'T':
        tpp_hw /= 1000
        mem_bw_hw /= 1000

    # Define the subplot
    ax = plt.subplot(1, 1, 1)
    plt.title(hw, fontsize=(12 if 'NPU' not in hw else 10))
    plt.xlabel('Arithmetic Intensity (FLOP/Byte)')
    plt.ylabel(f'Performance ({scale}FLOP/s)')

    # Plot the compute boundaries
    plt.axhline(y=tpp_hw, color='gray', linestyle='dashed')

    # Plot the data points
    flops_byte = [flop / mem for flop, mem in zip(data[_kp+'flops_avg'], data[_kp+'memory_avg'])]
    plt.scatter(flops_byte, data_gflops, color=[colors[models[name]-1] for name in data['model.name']], alpha=0.8)


    texts = []
    for j, (x, y, name) in enumerate(zip(flops_byte, data_gflops, data['model.name'])):
        mid = models[name]
        texts.append(axs.text(x, y, f"{mid}", fontsize=8, color=colors[mid-1], ha='center', va='center'))
    # adjust_text(texts)
    adjust_text(texts, force_text=(0.2, 0.2))

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


plt.tight_layout(pad=0.1)

# Save the figure
plt.savefig(args.output_file_stem + '.png', dpi=240)
plt.savefig(args.output_file_stem + '.pdf')
