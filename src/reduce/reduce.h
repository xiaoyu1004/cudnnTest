#ifndef REDUCE_H
#define REDUCE_H

template <typename T>
void reduceMax4D(unsigned int dim_n,
                 unsigned int dim_c,
                 unsigned int dim_h,
                 unsigned int dim_w,
                 T *workspace,
                 const T *A,
                 T *C);

template <typename T>
void reduceMax5D(unsigned int dim_n,
                 unsigned int dim_c,
                 unsigned int dim_d,
                 unsigned int dim_h,
                 unsigned int dim_w,
                 T *workspace,
                 const T *A,
                 T *C);

#endif