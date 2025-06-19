#include <ATen/ATen.h> 
#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>  
#include <c10/util/BFloat16.h> 
#include <c10/util/Exception.h> 

#include <iostream>
#include <typeinfo>
#include <cstdio> 

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define _KERNEL_DISPATCH_CASE(SCALAR_TYPE_ENUM_VAL, CXX_SCALAR_TYPE, LAMBDA_FN_TO_CALL) \
  case SCALAR_TYPE_ENUM_VAL: {                                                               \
    using scalar_t = CXX_SCALAR_TYPE;                 \
    LAMBDA_FN_TO_CALL();                                       \
    break;                                                                                   \
  }

#define _KERNEL_FLOATING_CASES(LAMBDA_FN_TO_CALL)                                                  \
  _KERNEL_DISPATCH_CASE(at::ScalarType::Float, float, LAMBDA_FN_TO_CALL)                           \
  _KERNEL_DISPATCH_CASE(at::ScalarType::Half, at::Half, LAMBDA_FN_TO_CALL)                         \
  _KERNEL_DISPATCH_CASE(at::ScalarType::BFloat16, at::BFloat16, LAMBDA_FN_TO_CALL)                 \
  _KERNEL_DISPATCH_CASE(at::ScalarType::Double, double, LAMBDA_FN_TO_CALL)


#define DISPATCH_KERNEL_BY_TYPE(SCALAR_TYPE_ENUM_INPUT, KERNEL_NAME_STR, LAMBDA_FN) \
  do { \
    switch (SCALAR_TYPE_ENUM_INPUT) { \
      _KERNEL_FLOATING_CASES(LAMBDA_FN) \
      default: \
        AT_ERROR( \
            KERNEL_NAME_STR, \
            " kernel does not support Pytorch ScalarType: '", \
            c10::toString(SCALAR_TYPE_ENUM_INPUT), \
            "'"); \
    } \
  } while (0)
