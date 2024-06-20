import os
import sys
import argparse
import traceback
import subprocess

parser = argparse.ArgumentParser(
        description='export single model in timm to onnx',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-l', "--list", action='store_true', help='list all model')
parser.add_argument("--flops", action='store_true', help='get flops (macs * 2) and exit')
parser.add_argument('model_name', nargs='?', help='model name')
parser.add_argument('-t', "--test_with_trtexec", action='store_true', help='test with trtexec in tensorrt')
parser.add_argument('--bs1', action='store_true', help='batch_size=1 instead of dynamic')
parser.add_argument('-o', "--opset", type=int, default=13, help='ONNX opset version')
args = parser.parse_args()


import timm
all_models = timm.list_models(pretrained=True)

if args.list:
    print(*all_models, sep='\n')

model_name = args.model_name or input("input model name: ")

if args.bs1:
    dyn = {}
else:
    dyn = {'input' : {0 : 'batch_size'},
           'output' : {0 : 'batch_size'}}

import torch

SAVE_DIR = "./"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
SKIP_EXIST = False
OPSET_VERSION = args.opset

dummy_input_224 = torch.zeros(1, 3, 224, 224)


print(" + export:", model_name)

onnx_path = SAVE_DIR+'/'+model_name+".onnx"
if os.path.isfile(onnx_path):
    if SKIP_EXIST:
        print("   model exist, skip")
        sys.exit(0)

try:
    torch_model = timm.create_model(model_name, pretrained=True)
    torch_model.eval()

    # image size
    if hasattr(torch_model, "patch_embed"):
        x = torch.zeros(1, 3, *torch_model.patch_embed.img_size)
    else:
        x = dummy_input_224

    if args.flops:
        from thop import profile
        macs, params = profile(torch_model, inputs=(x, ))
        print('mflops', 'params', macs / 1e6 * 2, params)

    # continue # uncomment to download only
    torch.onnx.export(torch_model,               # model being run
                    x,                           # model input (or a tuple for multiple inputs)
                    onnx_path,                   # where to save the model (can be a file or file-like object)
                    export_params=True,          # store the trained parameter weights inside the model file
                    opset_version=OPSET_VERSION, # the ONNX version to export the model to
                    do_constant_folding=True,    # whether to execute constant folding for optimization
                    input_names = ['input'],     # the model's input names
                    output_names = ['output'],   # the model's output names
                    dynamic_axes = dyn)

    if args.test_with_trtexec:
        print("   run trtexec")
        trt_command = [TRTEXEC, "--onnx="+onnx_path, "--saveEngine=tmp/test.trt"]
        result = subprocess.run(trt_command,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("   trtexec ok")
        else:
            os.unlink(onnx_path)
            print("   trtexec error")
            print(result.stdout)
            print(result.stderr)
except Exception as e:
    print("   other error", e)
    traceback.print_exc()
