import onnx
from onnx import IR_VERSION, TensorProto, __version__, shape_inference
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

print(
    f"onnx.__version__={__version__!r}, opset={onnx_opset_version()}, IR_VERSION={IR_VERSION}"
)


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)


# onnx_model = onnx.load("unet\model.onnx")
# onnx.checker.check_model(onnx_model)

# Preprocessing: load the ONNX model
model_path = "diffusion_onnx\diffusion.onnx"
onnx_model = onnx.load(model_path)
print(f"The model is:\n{model_path}")

# Check the model
# try:
#    onnx.checker.check_model(onnx_model)
# except onnx.checker.ValidationError as e:
#    print(f"The model is invalid: {e}")
# else:
#    print("The model is valid!")


# the list of inputs
print("** inputs **")
print(onnx_model.graph.input)

# in a more nicely format
print("** inputs **")
for obj in onnx_model.graph.input:
    print(
        "name=%r dtype=%r shape=%r"
        % (
            obj.name,
            obj.type.tensor_type.elem_type,
            shape2tuple(obj.type.tensor_type.shape),
        )
    )

# the list of outputs
print("** outputs **")
print(onnx_model.graph.output)

# in a more nicely format
print("** outputs **")
for obj in onnx_model.graph.output:
    print(
        "name=%r dtype=%r shape=%r"
        % (
            obj.name,
            obj.type.tensor_type.elem_type,
            shape2tuple(obj.type.tensor_type.shape),
        )
    )


print(
    "Before shape inference, the shape info of Y is:\n{}".format(
        onnx_model.graph.value_info
    )
)
inferred_model = shape_inference.infer_shapes(onnx_model)
onnx.shape_inference.infer_shapes_path(
    ".\diffusion_onnx\diffusion.onnx", ".\diffusion_onnx\diffusion.onnx"
)
onnx_model = onnx.load(".\diffusion_onnx\diffusion.onnx")
# print("After shape inference, the shape info of Y is:\n{}".format(inferred_model.graph.node.value_info))


# the list of nodes
print("** nodes **")
# print(onnx_model.graph.node)

# in a more nicely format
print("** nodes **")
for node in onnx_model.graph.node:
    print(
        "name=%r type=%r input=%r output=%r"
        % (node.name, node.op_type, node.input, node.output)
    )
    try:
        print(onnx.graph.node.value_info)
    except:
        pass
    if node.op_type == "Conv":
        input("Enter a key")
