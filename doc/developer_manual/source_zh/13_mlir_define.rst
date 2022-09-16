MLIR定义
============

本章介绍MLIR各个元素的定义，包括Dialect、Interface等等

Top Dialect
---------------

Operations
~~~~~~~~~~~~~~~

AddOp
^^^^^^^^^^^^^^^

:简述:
    加法操作，:math:`Y = coeff_0 * X_0 + coeff_1 * X_1`

:输入:
    - inputs: tensor数组，对应2个或多个输入tensor

:输出:
    - output: tensor

:属性:
    - do_relu: 结果是否做Relu，默认为False
    - relu_limit: 如果做Relu，指定上限值，如果是负数，则认为没有上限
    - coeff: 对应每个tensor的系数，默认为1.0

:输出:
    - output: 输出tensor

:接口:
    无

:范例:
    .. code-block:: console

      %2 = "tpu.Add"(%0, %1) {do_relu = false} : (tensor<1x3x27x27xf32>, tensor<1x3x27x27xf32>) -> tensor<1x3x27x27xf32> loc("add")


AvgPoolOp
^^^^^^^^^^^^^^^
(待补充)

Depth2SpaceOp
^^^^^^^^^^^^^^^
(待补充)

BatchNormOp
^^^^^^^^^^^^^^^
(待补充)

CastOp
^^^^^^^^^^^^^^^
(待补充)

ClipOp
^^^^^^^^^^^^^^^
(待补充)

ConcatOp
^^^^^^^^^^^^^^^
(待补充)

ConvOp
^^^^^^^^^^^^^^^
(待补充)

DeconvOp
^^^^^^^^^^^^^^^
(待补充)

DivOp
^^^^^^^^^^^^^^^
(待补充)

InputOp
^^^^^^^^^^^^^^^
(待补充)

LeakyReluOp
^^^^^^^^^^^^^^^
(待补充)

LSTMOp
^^^^^^^^^^^^^^^
(待补充)

LogOp
^^^^^^^^^^^^^^^
(待补充)

MaxPoolOp
^^^^^^^^^^^^^^^
(待补充)

MatMulOp
^^^^^^^^^^^^^^^
(待补充)

MulOp
^^^^^^^^^^^^^^^
(待补充)

MulConstOp
^^^^^^^^^^^^^^^
(待补充)

PermuteOp
^^^^^^^^^^^^^^^
:简述:
    改变tensor布局:变化tensor数据维度的顺序，将输入的tensor按照order给定的顺序重新布局

:输入:
    - inputs: tensor数组，任意类型的tensor


:属性:
    - order: 指定重新布局tensor的顺序


:输出:
    - output: 输出tensor，按order的顺序重新布局后的tensor

:接口:
    无

:范例:
    .. code-block:: console

      %2 = "top.Permute"(%1) {order = [0, 1, 3, 4, 2]} : (tensor<4x3x85x20x20xf32>) -> tensor<4x3x20x20x85xf32> loc("output_Transpose")



ReluOp
^^^^^^^^^^^^^^^
(待补充)

ReshapeOp
^^^^^^^^^^^^^^^
(待补充)

ScaleOp
^^^^^^^^^^^^^^^
(待补充)

SigmoidOp
^^^^^^^^^^^^^^^
:简述:
    激活函数，将tensor中元素映射到特定区间，默认映射到[0，1]，:math:`Y = scale / (1 + exp(-X)) + bias` 

:输入:
    - inputs: tensor数组，任意类型的tensor


:属性:
    - scale: 倍数，默认是1
    - bias: 偏置，默认是0


:输出:
    - output: 输出tensor

:接口:
    无

:范例:
    .. code-block:: console

      %2 = "top.Sigmoid"(%1) {bias = 0.000000e+00 : f64, scale = 1.000000e+00 : f64} : (tensor<1x16x64x64xf32>) -> tensor<1x16x64x64xf32> loc("output_Sigmoid")



SiLUOp
^^^^^^^^^^^^^^^
:简述:
    激活函数，:math:`Y = X / (1 + exp(-X))` 或 :math:`Y = X * Sigmoid(X)`

:输入:
    - input: tensor数组，任意类型的tensor


:属性:
    无


:输出:
    - output: 输出tensor

:接口:
    无

:范例:
    .. code-block:: console

        %1 = "top.SiLU"(%0) : (tensor<1x16x64x64xf32>) -> tensor<1x16x64x64xf32> loc("output_Mul")



SliceOp
^^^^^^^^^^^^^^^
:简述: tensor切片，将输入的tensor的各个维度，根据offset和steps数组中的偏移和步长进行切片，生成新的tesnor
    

:输入:
    - input: tensor数组，任意类型的tensor


:属性:
    - offset: 存储切片偏移的数组，offset数组的索引和输入tensor的维度索引对应
    - steps: 存储切片步长的数组，steps数组的索引和输入tensor维度索引对应


:输出:
    - output: 输出tensor

:接口:
    无

:范例:
    .. code-block:: console

        %1 = "top.Slice"(%0) {offset = [2, 10, 10, 12], steps = [1, 2, 2, 3]} : (tensor<5x116x64x64xf32>) -> tensor<3x16x16x8xf32> loc("output_Slice")




SoftmaxOp
^^^^^^^^^^^^^^^
:简述:
    对输入tensor，在指定axis的维度上计算归一化指数值， :math:`Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)`
    
    其中， :math:`ReduceSum(Exp(input), axis=axis, keepdims=1)` 表示在axis维度上计算Exp(input)的和，得到的tensor维度和输入tensor维度一致。 
:输入:
    - input: tensor数组，任意类型的tensor


:属性:
    - axis: 维度索引，用于指定对输入tensor执行Softmax对应的维度，axis可以取值[-r， r-1], r 为输入tensor维度的数量, 当axis为负数时，表示倒序维度
    - beta: tflite模型中对输入的缩放系数，非tflite模型无效，默认值为1.0


:输出:
    - output: 输出tensor，在指定维度做归一化指数值后的tensor

:接口:
    无

:范例:
    .. code-block:: console

      %1 = "top.Softmax"(%0) {axis = 1 : i64} : (tensor<1x1000x1x1xf32>) -> tensor<1x1000x1x1xf32> loc("output_Softmax")


SqueezeOp
^^^^^^^^^^^^^^^
(待补充)

UpsampleOp
^^^^^^^^^^^^^^^
(待补充)

WeightOp
^^^^^^^^^^^^^^^

:简述:
    权重op，包括权重的读取和创建，权重会存到npz文件中。权重的location与npz中的tensor名称是对应关系。

:输入:
    无

:属性:
    无

:输出:
    - output: 权重Tensor

:接口:
    - read: 读取权重数据，类型由模型指定
    - read_as_float: 将权重数据转换成float类型读取
    - read_as_byte: 将权重数据按字节类型读取
    - create: 创建权重op
    - clone_bf16: 将当前权重转换成bf16，并创建权重Op
    - clone_f16: 将当前权重转换成f16，并创建权重Op

:范例:
    .. code-block:: console

      %1 = "top.Weight"() : () -> tensor<32x16x3x3xf32> loc("filter")

