"""
ONNX Inspector Script

Dissects the exported ONNX model to reveal its internal node structure,
verifying the INT8 quantization and the exact operations that survived
the PyTorch to ONNX translation.
"""

from pathlib import Path

import onnx


def inspect_onnx_model(onnx_path: str):
    path = Path(onnx_path)
    if not path.exists():
        print(f"Error: Could not find {path}")
        return

    print("==================================================")
    print(f"Dissecting ONNX Model: {path.name} ({path.stat().st_size / 1024:.2f} KB)")
    print("==================================================")

    # Load the model
    model = onnx.load(str(path))
    graph = model.graph

    # 1. Inputs
    print("\n[INPUTS]")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  - Name: '{inp.name}' | Type: Tensor | Shape: {shape}")

    # 2. Outputs
    print("\n[OUTPUTS]")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  - Name: '{out.name}' | Type: Tensor | Shape: {shape}")

    # 3. Node Inventory
    print("\n[NODE INVENTORY]")
    op_counts = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    # Print distinct operations sorted by frequency
    sorted_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)
    for op, count in sorted_ops:
        print(f"  - {op.ljust(25)}: {count} ops")

    # 4. Deep Dive into Execution Path
    print("\n[EXECUTION PATH PREVIEW (First 15 Nodes)]")
    for i, node in enumerate(graph.node[:15]):
        inputs = [i for i in node.input if i] # Filter empty strings
        outputs = list(node.output)
        print(f"  {i+1:02d}. {node.op_type}")
        print(f"      In:  {inputs}")
        print(f"      Out: {outputs}")

    if len(graph.node) > 15:
        print(f"  ... and {len(graph.node) - 15} more nodes.")

    print(f"\nInspection Complete. Total Nodes: {len(graph.node)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deployment/onnx/hybrid_v35_int8.onnx")
    args = parser.parse_args()

    inspect_onnx_model(args.model)
