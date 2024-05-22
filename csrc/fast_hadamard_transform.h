/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct HadamardParamsBase {
    using index_t = int64_t;

    int batch, dim, log_N;
    bool scale_first;

    index_t x_batch_stride;
    index_t scales_batch_stride;
    index_t out_batch_stride;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ scales_ptr;
    void *__restrict__ out_ptr;
};
