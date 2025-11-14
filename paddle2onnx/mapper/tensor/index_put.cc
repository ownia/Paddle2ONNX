#include "paddle2onnx/mapper/tensor/index_put.h"

namespace paddle2onnx {
REGISTER_MAPPER(index_put, IndexPutMapper)
REGISTER_PIR_MAPPER(index_put, IndexPutMapper)

int32_t IndexPutMapper::GetMinOpsetVersion(bool verbose) {
  Logger(verbose, 11) << RequireOpset(11) << std::endl;
  return 11;
}

void IndexPutMapper::Opset11() {
  auto x_info = GetInput("X");
  auto indices_info = GetInput("Indices");
  auto updates_info = GetInput("Value");

  auto out_info = GetOutput("Out");

  std::vector<int64_t> perm = {1, 0};
  auto transpose_node = helper_->MakeNode("Transpose", {indices_info[0].name});
  AddAttribute(transpose_node, "perm", perm);

  auto node = helper_->MakeNode(
      "ScatterND",
      {x_info[0].name, transpose_node->output(0), updates_info[0].name},
      {out_info[0].name}
  );
  //AddAttribute(node, "axis", axis_);
}

}  // namespace paddle2onnx
