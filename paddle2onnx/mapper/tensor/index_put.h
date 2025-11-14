#pragma once
#include <string>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class IndexPutMapper : public Mapper {
 public:
  IndexPutMapper(const PaddleParser& p,
                    OnnxHelper* helper,
                    int64_t block_id,
                    int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {}
  IndexPutMapper(const PaddlePirParser& p,
                    OnnxHelper* helper,
                    int64_t op_id,
                    bool c)
      : Mapper(p, helper, op_id, c) {
    in_pir_mode = true;
  }
  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset11() override;

 private:
  int64_t axis_;
};

}  // namespace paddle2onnx
