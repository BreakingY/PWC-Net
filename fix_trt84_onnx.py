import onnx


src_path = "pwcnet_trt84.onnx"
dst_path = "pwcnet_trt84_fixed.onnx"

model = onnx.load(src_path)


# ============================================================
# 1. 建立 Identity 映射
#
# Identity:
#
#   input  -> output
#
# 例如：
#
#   model.netTwo.netSix.0.bias
#             |
#          Identity
#             |
#   model.netRefiner.netMain.12.bias
# ============================================================

identity_map = {}

for node in model.graph.node:

    if node.op_type != "Identity":
        continue

    if len(node.input) != 1 or len(node.output) != 1:
        continue

    identity_map[node.output[0]] = node.input[0]


# ============================================================
# 2. 获取 initializer
# ============================================================

initializer_names = {
    initializer.name
    for initializer in model.graph.initializer
}


print("=" * 80)
print("ONNX TRT 8.4 Compatibility Fix")
print("=" * 80)


# ============================================================
# 3. Fix ConvTranspose bias
# ============================================================

base_bias = "model.netTwo.netUpflow.bias"

count_convtranspose = 0


print()
print("=" * 80)
print("Fix ConvTranspose bias")
print("=" * 80)


for node in model.graph.node:

    if node.op_type != "ConvTranspose":
        continue

    if len(node.input) < 3:
        continue

    old_bias = node.input[2]

    # Identity -> shared bias
    source_bias = identity_map.get(old_bias)

    if source_bias == base_bias:

        print()
        print("Node:", node.name)
        print("  old bias:", old_bias)
        print("  new bias:", base_bias)

        node.input[2] = base_bias

        count_convtranspose += 1


# ============================================================
# 4. Fix Conv bias
#
# TRT 8.4 要求：
#
#   Conv bias 必须是 initializer
#
# 如果发现：
#
#   Conv -> bias -> Identity -> initializer
#
# 则直接把 Conv 的 bias 改成真正的 initializer。
# ============================================================

count_conv = 0


print()
print("=" * 80)
print("Fix Conv bias")
print("=" * 80)


for node in model.graph.node:

    if node.op_type != "Conv":
        continue

    if len(node.input) < 3:
        continue

    old_bias = node.input[2]

    # 已经是 initializer，不需要处理
    if old_bias in initializer_names:
        continue

    # 查找 Identity 的源头
    source_bias = identity_map.get(old_bias)

    if source_bias is None:
        continue

    # Identity 的源头必须是 initializer
    if source_bias not in initializer_names:
        continue

    print()
    print("Node:", node.name)
    print("  old bias:", old_bias)
    print("  source  :", source_bias)
    print("  status  : Identity -> initializer")
    print("  fix     :", old_bias, "=>", source_bias)

    node.input[2] = source_bias

    count_conv += 1


# ============================================================
# 5. 检查所有 Conv / ConvTranspose bias
# ============================================================

print()
print("=" * 80)
print("Final Conv / ConvTranspose bias check")
print("=" * 80)


bad_biases = []


for node in model.graph.node:

    if node.op_type not in ("Conv", "ConvTranspose"):
        continue

    if len(node.input) < 3:
        continue

    bias = node.input[2]

    if bias not in initializer_names:

        bad_biases.append(
            (
                node.op_type,
                node.name,
                bias,
            )
        )

        print()
        print("[WARNING]")
        print("  op   :", node.op_type)
        print("  node :", node.name)
        print("  bias :", bias)
        print("  !!! bias is NOT initializer")


# ============================================================
# 6. 删除已经不再使用的 Identity
#
# 注意：
# 这里只删除我们修复后已经没有任何消费者的 Identity。
# ============================================================

used_inputs = set()

for node in model.graph.node:
    for inp in node.input:
        used_inputs.add(inp)


remove_identity_nodes = []


for node in model.graph.node:

    if node.op_type != "Identity":
        continue

    if len(node.output) != 1:
        continue

    output_name = node.output[0]

    # 如果 Identity 输出已经没有任何节点使用
    if output_name not in used_inputs:

        remove_identity_nodes.append(node)


for node in remove_identity_nodes:

    print()
    print("Remove unused Identity:")
    print("  node  :", node.name)
    print("  input :", node.input[0])
    print("  output:", node.output[0])

    model.graph.node.remove(node)


# ============================================================
# 7. 保存
# ============================================================

print()
print("=" * 80)
print("结果")
print("=" * 80)

print("修改 ConvTranspose 数量:", count_convtranspose)
print("修改 Conv bias 数量     :", count_conv)
print("剩余非 initializer bias :", len(bad_biases))


if bad_biases:

    print()
    print("!!! 仍然存在不符合 TRT 8.4 要求的 bias:")
    for op_type, node_name, bias in bad_biases:
        print(" ", op_type, node_name, bias)

else:

    print()
    print("✅ 所有 Conv / ConvTranspose bias 都是 initializer")


onnx.save(model, dst_path)

print()
print("保存到:", dst_path)
print("=" * 80)
