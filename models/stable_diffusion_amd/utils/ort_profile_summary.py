import argparse
import json


def getLogData(filename):
    print(f"---- Started Reading Log JSON file {filename}")
    f = open(filename, "r")
    print("---- Reading JSON encoded data.")
    return json.load(f)


def onnxSummary(logData):
    cpuTimeDict = {}
    ipuTimeDict = {}
    cpuFreqDict = {}
    ipuFreqDict = {}
    cpuOpsTotalTime = {}
    for ld in logData:
        if ld["cat"] == "Node" and ld["dur"] > 0 and "provider" in ld["args"]:
            op_name = ld["args"]["op_name"]
            # print(op_name)
            # print(ld['dur'])
            input_type = (
                str(ld["args"]["input_type_shape"])
                if "input_type_shape" in ld["args"]
                else ""
            )
            # print(input_type)
            provider = ld["args"]["provider"]
            # print(provider)
            if provider == "CPUExecutionProvider":
                if op_name not in cpuTimeDict:
                    cpuTimeDict[op_name] = {}
                    cpuFreqDict[op_name] = {}
                if input_type not in cpuTimeDict[op_name]:
                    cpuTimeDict[op_name][input_type] = 0
                    cpuFreqDict[op_name][input_type] = 0
                cpuTimeDict[op_name][input_type] += ld["dur"]
                cpuFreqDict[op_name][input_type] += 1

                if op_name not in cpuOpsTotalTime:
                    cpuOpsTotalTime[op_name] = 0
                cpuOpsTotalTime[op_name] += ld["dur"]
            else:
                if input_type not in ipuTimeDict:
                    ipuTimeDict[input_type] = 0
                    ipuFreqDict[input_type] = 0
                ipuTimeDict[input_type] += ld["dur"]
                ipuFreqDict[input_type] += 1

    # print (cpuOpsTotalTime, ipuOpsTotalTime)
    cpuOps = sorted(cpuOpsTotalTime, key=lambda k: cpuOpsTotalTime[k], reverse=True)
    ipuOpsShape = sorted(ipuTimeDict, key=lambda k: ipuTimeDict[k], reverse=True)
    # print (cpuOps, ipuOps)

    cpu_csv_lines = []
    ipu_csv_lines = []
    print("===IPU Provider Time Breakdown:")

    print("  ", "  ", "Total: ", f"{sum(ipuTimeDict.values())*1e-6}s")
    for inp in ipuTimeDict:
        print("  ", "  ", f"{inp: <25}: ", f"{ipuTimeDict[inp]}us")
        l = (
            "ipu;"
            + inp
            + ";"
            + str(ipuTimeDict[inp])
            + ";"
            + str(ipuFreqDict[inp])
            + "\n"
        )
        ipu_csv_lines.append(l)
    print("\n===CPU Provider Time Breakdown:")
    for op in cpuOps:
        print("  ", op)
        print("  ", "  ", "Total: ", f"{sum(cpuTimeDict[op].values())*1e-6}s")
        for inp in cpuTimeDict[op]:
            print("  ", "  ", f"{inp: <25}: ", f"{cpuTimeDict[op][inp]}us")
            l = (
                op
                + ";"
                + inp
                + ";"
                + str(cpuTimeDict[op][inp])
                + ";"
                + str(cpuFreqDict[op][inp])
                + "\n"
            )
            cpu_csv_lines.append(l)

    with open("ipu_ops_profile.csv", "w") as f:
        f.writelines(ipu_csv_lines)

    with open("cpu_ops_profile.csv", "w") as f:
        f.writelines(cpu_csv_lines)

    # print("CPU Ops Summary:\n", f"{'Total:': <27}", sum(cpu_ops.values()))
    # for op in sorted(cpu_ops.keys()):
    #     print ("  ", f"{op: <25}", cpu_ops[op])
    # print("\nDPU Ops Summary:\n", f"{'Total:': <27}", sum(dpu_ops.values()))
    # for op in sorted(dpu_ops.keys()):
    #     print ("  ", f"{op: <25}", dpu_ops[op])
    # print("\nGEMM Ops Summary:\n", f"{'Total:': <27}", sum(gemm_ops.values()))
    # for op in sorted(gemm_ops.keys()):
    #     print ("  ", f"{op: <25}", gemm_ops[op])

    # all_ops = set(cpu_ops.keys()).union(set(dpu_ops.keys())).union(set(gemm_ops.keys()))
    # print("\nOps Breakdown:\n", f"  {'Op_Type': <25}", f"{'CPU': <10}", f"{'DPU': <10}", f"{'GEMM': <10}")
    # for op in sorted(list(all_ops)):
    #     cpu_op_count = cpu_ops[op] if op in cpu_ops else 0
    #     dpu_op_count = dpu_ops[op] if op in dpu_ops else 0
    #     gemm_op_count = gemm_ops[op] if op in gemm_ops else 0
    #     total = cpu_op_count + dpu_op_count + gemm_op_count
    #     print ("  ", f"{op: <25}", f"{cpu_op_count/float(total): <10.2%}", f"{dpu_op_count/float(total): <10.2%}", f"{gemm_op_count/float(total): <10.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", help="Log files ", nargs="+", default=["onnxruntime_profile.json"]
    )
    args = parser.parse_args()
    logData = []
    for f in args.files:
        logData += getLogData(f)

    onnxSummary(logData)
