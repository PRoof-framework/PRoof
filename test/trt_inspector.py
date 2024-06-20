import argparse
import json
import tensorrt as trt

parser = argparse.ArgumentParser(
        description='test tensorrt by run a engine file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('engine', help='engine file')
args = parser.parse_args()


logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

with open(args.engine, 'rb') as f:
    serialized_engine = f.read()
engine = runtime.deserialize_cuda_engine(serialized_engine)

print(engine)
context = engine.create_execution_context()

trt.EngineInspector.engine = engine
trt.EngineInspector.context = context

inspector = engine.create_engine_inspector()
inspector.execution_context = context
engine_info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
print(json.dumps(engine_info, indent=2))
print("%d layers" % len(engine_info['Layers']))

# print(inspector.get_layer_information(0, trt.LayerInformationFormat.JSON)) # Print the information of the first layer in the engine.

