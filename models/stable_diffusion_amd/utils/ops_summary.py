import argparse
import json


def getLogData(filename):
    print(f"---- Started Reading Log JSON file {filename}")
    f = open(filename, "r")
    print("---- Reading JSON encoded data.")
    return json.load(f)["nodeStat"]


def opsSummary(logData):
    cpu_ops = {}
    dpu_ops = {}
    gemm_ops = {}

    for ns in logData:
        if ns["device"] == "CPU":
            if ns["opType"] not in cpu_ops:
                cpu_ops[ns["opType"]] = 0
            cpu_ops[ns["opType"]] += 1
        elif ns["device"] == "DPU":
            if ns["opType"] not in dpu_ops:
                dpu_ops[ns["opType"]] = 0
            dpu_ops[ns["opType"]] += 1
        elif ns["device"] == "GEMM":
            if ns["opType"] not in gemm_ops:
                gemm_ops[ns["opType"]] = 0
            gemm_ops[ns["opType"]] += 1

    print("CPU Ops Summary:\n", f"{'Total:': <27}", sum(cpu_ops.values()))
    for op in sorted(cpu_ops.keys()):
        print("  ", f"{op: <25}", cpu_ops[op])
    print("\nDPU Ops Summary:\n", f"{'Total:': <27}", sum(dpu_ops.values()))
    for op in sorted(dpu_ops.keys()):
        print("  ", f"{op: <25}", dpu_ops[op])
    print("\nGEMM Ops Summary:\n", f"{'Total:': <27}", sum(gemm_ops.values()))
    for op in sorted(gemm_ops.keys()):
        print("  ", f"{op: <25}", gemm_ops[op])

    all_ops = set(cpu_ops.keys()).union(set(dpu_ops.keys())).union(set(gemm_ops.keys()))
    print(
        "\nOps Breakdown:\n",
        f"  {'Op_Type': <25}",
        f"{'CPU': <10}",
        f"{'DPU': <10}",
        f"{'GEMM': <10}",
    )
    for op in sorted(list(all_ops)):
        cpu_op_count = cpu_ops[op] if op in cpu_ops else 0
        dpu_op_count = dpu_ops[op] if op in dpu_ops else 0
        gemm_op_count = gemm_ops[op] if op in gemm_ops else 0
        total = cpu_op_count + dpu_op_count + gemm_op_count
        print(
            "  ",
            f"{op: <25}",
            f"{cpu_op_count/float(total): <10.2%}",
            f"{dpu_op_count/float(total): <10.2%}",
            f"{gemm_op_count/float(total): <10.2%}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", help="Log files ", nargs="+", default=["vitisai_ep_report.json"]
    )
    args = parser.parse_args()
    logData = []
    for f in args.files:
        logData += getLogData(f)

    opsSummary(logData)
