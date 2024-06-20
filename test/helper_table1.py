import argparse

import pandas as pd

parser = argparse.ArgumentParser(
    prog = "image render",
    description = "",
    epilog = "WIP",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("ncu_csv_file", help="")
parser.add_argument("nsys_csv_file", help="")
args = parser.parse_args()


data_ncu = pd.read_csv(args.ncu_csv_file)
data_nsys = pd.read_csv(args.nsys_csv_file)

# print(data_ncu)
# print(data_nsys)

print("model name".ljust(35) + "latency (ms)".rjust(15) + "Predict GFLOP".rjust(20) + "Predict Memory (MB)".rjust(22) + "NCU GFLOP".rjust(20) + "NCU Memory (MB)".rjust(22) + "FLOP diff.".rjust(15) + "Memory diff.".rjust(15))
print("-"*166)
_kp = 'model.bench.results.*.'
for i in range(5):
    assert data_ncu['model.name'][i] == data_nsys['model.name'][i]
    name = data_ncu['model.name'][i]
    p_latency = data_nsys[_kp+'time_avg'][i]
    n_latency = data_ncu[_kp+'time_avg'][i]
    p_gflop = data_nsys[_kp+'flops_avg'][i] / 1e9 * p_latency
    n_gflop = data_ncu[_kp+'flops_avg'][i] / 1e9 * n_latency
    p_mem = data_nsys[_kp+'memory_avg'][i] / 1e6 * p_latency
    n_mem = data_ncu[_kp+'memory_avg'][i] / 1e6 * n_latency

    diff_gflop = f"{(p_gflop - n_gflop) / n_gflop * 100:.2f} %"
    diff_mem = f"{(p_mem - n_mem) / n_mem * 100:.2f} %"

    print(f"{name:<35}{n_latency*1e3:>15.3f}{p_gflop:>20.3f}{p_mem:>22.3f}{n_gflop:>20.3f}{n_mem:>22.3f}{diff_gflop:>15}{diff_mem:>15}")

