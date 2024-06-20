import os
import sys
import traceback
import subprocess

import timm
import torch

all_models = timm.list_models(pretrained=True)
# print(*all_models, sep='\n')

SAVE_DIR = "./tmp/timm_onnx"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
SKIP_EXIST = True
START_INDEX = 0
SHUFFLE = True

if SHUFFLE:
    import random
    random.shuffle(all_models)


dummy_input_224 = torch.zeros(1, 3, 224, 224)
for i, model_name in enumerate(all_models[START_INDEX:]):
    print(" + export", i, ":", model_name)

    onnx_path = SAVE_DIR+'/'+model_name+".onnx"
    if os.path.isfile(onnx_path):
        if SKIP_EXIST:
            print("   model exist, skip")
            continue

    try:
        torch_model = timm.create_model(model_name, pretrained=True)
        torch_model.eval()

        # image size
        if hasattr(torch_model, "patch_embed"):
            x = torch.zeros(1, 3, *torch_model.patch_embed.img_size)
        else:
            x = dummy_input_224

        # continue # uncomment to download only
        torch.onnx.export(torch_model,             # model being run
                        x,                         # model input (or a tuple for multiple inputs)
                        onnx_path,                 # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=13,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                        'output' : {0 : 'batch_size'}})

        print("   run trtexec")
        trt_command = [TRTEXEC, "--onnx="+onnx_path, "--saveEngine=tmp/test.trt"]
        result = subprocess.run(trt_command,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("   ok")
        else:
            os.unlink(onnx_path)
            print("   trtexec error")
            print(result.stdout)
            print(result.stderr)
    except Exception as e:
        print("   other error", e)
        traceback.print_exc()
