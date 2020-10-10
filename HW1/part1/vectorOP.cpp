#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones(N - i);

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //

  __pp_vec_float val, out, clamp;
  __pp_vec_int exp, i_zeros, i_ones;
  __pp_mask maskAll, maskValid, maskExpZero, maskTmp;

  // Initialize masks.
  maskAll = _pp_init_ones();

  // Initlaize vector registers values.
  _pp_vset_int(i_zeros, 0, maskAll);
  _pp_vset_int(i_ones, 1, maskAll);
  _pp_vset_float(clamp, 9.999999f, maskAll);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // Initialize masks.
    maskValid = _pp_init_ones(N - i);
    maskExpZero = _pp_init_ones(0);

    // Load values.
    _pp_vload_float(val, values + i, maskAll);
    _pp_vload_float(out, values + i, maskAll);
    // Load exponents.
    _pp_vset_int(exp, 0, maskAll);
    _pp_vload_int(exp, exponents + i, maskValid);

    // Set the value of data with exponent as 0 to 1.f.
    _pp_veq_int(maskExpZero, exp, i_zeros, maskAll);
    _pp_vset_float(out, 1.f, maskExpZero);

    // Loop until all exponents become 0.
    _pp_vsub_int(exp, exp, i_ones, maskAll);
    _pp_vgt_int(maskExpZero, exp, i_zeros, maskAll);

    while (_pp_cntbits(maskExpZero)) {
      _pp_vmult_float(out, out, val, maskExpZero);
      _pp_vsub_int(exp, exp, i_ones, maskAll);
      _pp_vgt_int(maskExpZero, exp, i_zeros, maskAll);
    }
    _pp_vgt_float(maskTmp, out, clamp, maskValid);
    _pp_vset_float(out, 9.999999f, maskTmp);
    _pp_vstore_float(output + i, out, maskValid);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
  }

  return 0.0;
}
