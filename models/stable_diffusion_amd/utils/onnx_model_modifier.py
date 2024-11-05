import argparse

import onnx
from onnx import numpy_helper

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    help="Model path",
    type=str,
)


def opsetConvert(model, opsetVersion):
    # nodes_num = len(model.graph.node)
    # for i in range(nodes_num):
    #     n = model.graph.node[i]
    #     if n.op_type == "InstanceNormalization":
    #         e = n.attribute[0].f
    #         new_n = onnx.helper.make_node(
    #             "InstanceNormalization",
    #             n.input,
    #             n.output,
    #             n.name + "_new",
    #             epsilon=e,
    #         )
    #         # print(new_n)
    #         model.graph.node.remove(n)
    #         model.graph.node.insert(i, new_n)
    #     elif n.op_type == "Conv":
    #         new_n = onnx.helper.make_node(
    #             "Conv",
    #             n.input,
    #             n.output,
    #             n.name + "_new",
    #         )
    #         for a in n.attribute:
    #             new_n.attribute.append(a)
    #         # print(new_n)
    #         model.graph.node.remove(n)
    #         model.graph.node.insert(i, new_n)
    return onnx.version_converter.convert_version(model, opsetVersion)


def addCast2Conv(model):
    nodes_num = len(model.graph.node)
    newNodesList = []
    newNodesLen = 0

    for i in range(nodes_num):
        n = model.graph.node[i]

        if n.op_type == "Conv":
            input_x, input_w, input_b = n.input
            outputConv = n.output[0]
            iniNum = len(model.graph.initializer)
            for j in range(iniNum):
                ini = model.graph.initializer[j]
                if ini.name == input_w or ini.name == input_b:
                    print(ini.name)
                    newIni = numpy_helper.from_array(
                        numpy_helper.to_array(ini), ini.name + "_to_fp32"
                    )
                    model.graph.initializer.remove(ini)
                    model.graph.initializer.insert(j, newIni)
                    if ini.name == input_w:
                        n.input[1] = ini.name + "_to_fp32"
                    else:
                        n.input[2] = ini.name + "_to_fp32"
            cast1Out = input_x + "_cast_to_fp32_output"
            cast1 = onnx.helper.make_node(
                "Cast", [input_x], [cast1Out], n.name + "_cast2fp32", to=1
            )
            n.input[0] = cast1Out
            conv_new_output = outputConv + "_2castnode"
            n.output[0] = conv_new_output
            cast2 = onnx.helper.make_node(
                "Cast", [conv_new_output], [outputConv], n.name + "_cast2fp16", to=16
            )
            newNodesList.append(cast1)
            newNodesList.append(n)
            newNodesList.append(cast2)
            newNodesLen += 3
        else:
            newNodesList.append(n)
            newNodesLen += 1

    while len(model.graph.node):
        model.graph.node.pop()
    print(len(model.graph.node))
    for i in range(newNodesLen):
        model.graph.node.insert(i, newNodesList[i])


def in2gn(model):
    nodes_num = len(model.graph.node)
    for i in range(nodes_num):
        n = model.graph.node[i]
        if n.op_type == "InstanceNormalization":
            shape_inp = n.input[1]
            groups = 32
            for m in model.graph.node:
                if shape_inp in m.output:
                    groups = m.attribute[0].t.dims[0]
            e = n.attribute[0].f
            new_n = onnx.helper.make_node(
                "GroupNormalization",
                n.input,
                n.output,
                n.name + "_to_groupnorm",
                epsilon=e,
                num_groups=groups,
            )
            # print(new_n)
            model.graph.node.remove(n)
            model.graph.node.insert(i, new_n)


if __name__ == "__main__":
    args = parser.parse_args()
    model = onnx.load(args.model)
    # in2gn(model)
    # addCast2Conv(model)
    model = opsetConvert(model, 22)
    onnx.save(model, args.model)
