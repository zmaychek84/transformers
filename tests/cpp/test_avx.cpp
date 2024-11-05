#include "Eigen/Core"
#include "super_instr.h"
#include <algorithm>
#include <chrono>
#include <gtest/gtest.h>
#include <immintrin.h>
#include <iostream>
void fastMemcpy(void *pvDest, void *pvSrc, size_t nBytes) {

  const __m256i *pSrc = reinterpret_cast<const __m256i *>(pvSrc);
  __m256i *pDest = reinterpret_cast<__m256i *>(pvDest);
  int64_t nVects = nBytes / 32;
  for (; nVects > 0; nVects--, pSrc++, pDest++) {
    const __m256i loaded = _mm256_stream_load_si256(pSrc);
    _mm256_stream_si256(pDest, loaded);
  }
  _mm_sfence();
}
// Test for avx output tiling
TEST(avx, TILING) {
  int SEQ_BYTES = sizeof(ryzenai::GemmSeq);
  std::vector<int> kernel_x_shape = {8, 2048};
  std::vector<int> kernel_y_shape = {2048, 2048};
  std::tuple<int, int> a_shape = {8, 8192};
  std::tuple<int, int> b_shape = {2048, 8192};
  int64_t kernel_z_shape_[2];
  auto M = std::get<0>(a_shape);
  auto K = std::get<1>(a_shape);
  auto N = std::get<1>(b_shape);
  auto num_m_tiles_ = (int64_t)std::ceil((float)M / (float)kernel_x_shape[0]);
  auto num_k_tiles_ = (int64_t)std::ceil((float)K / (float)kernel_y_shape[0]);
  auto num_n_tiles_ = (int64_t)std::ceil((float)N / (float)kernel_y_shape[1]);
  kernel_z_shape_[0] = kernel_x_shape[0];
  kernel_z_shape_[1] = kernel_y_shape[1];
  int64_t C_BO_SIZE_TOKEN = num_m_tiles_ * num_k_tiles_ * kernel_z_shape_[1] *
                            num_n_tiles_ * kernel_z_shape_[0] * sizeof(int8_t);
  std::vector<int8_t> c_map;
  std::vector<int8_t> c_map_2;
  c_map.resize(C_BO_SIZE_TOKEN);

  for (int i = 0; i < C_BO_SIZE_TOKEN; i++)
    c_map[i] = i % 64;
  c_map_2.resize(C_BO_SIZE_TOKEN);
  for (int i = 0; i < C_BO_SIZE_TOKEN; i++)
    c_map_2[i] = i % 64;
  assert((kernel_z_shape_[1] % 64 == 0));
  // CPU
  auto t1 = std::chrono::steady_clock::now();
  for (int64_t m = 0; m < num_m_tiles_; m++) {
    for (int64_t n = 0; n < num_n_tiles_; n++) {
      for (int64_t k = 1; k < num_k_tiles_; k++) {
        auto dest_tile_idx =
            (m * num_k_tiles_ * num_n_tiles_ + n * num_k_tiles_) *
            kernel_z_shape_[0] * kernel_z_shape_[1];
        auto src_tile_idx =
            (m * num_k_tiles_ * num_n_tiles_ + n * num_k_tiles_ + k) *
            kernel_z_shape_[0] * kernel_z_shape_[1];

        for (int64_t i = 0; i < kernel_z_shape_[0]; i++) {
          for (int64_t j = 0; j < kernel_z_shape_[1]; j++) {
            auto dest_idx = dest_tile_idx + i * kernel_z_shape_[1] + j;
            auto src_idx = src_tile_idx + i * kernel_z_shape_[1] + j;
            c_map[dest_idx] += c_map[src_idx];
          }
        }
      }
    }
  }
  auto t2 = std::chrono::steady_clock::now();
  for (int64_t m = 0; m < num_m_tiles_; m++) {
    for (int64_t n = 0; n < num_n_tiles_; n++) {
      for (int64_t k = 1; k < num_k_tiles_; k++) {
        auto dest_tile_idx =
            (m * num_k_tiles_ * num_n_tiles_ + n * num_k_tiles_) *
            kernel_z_shape_[0] * kernel_z_shape_[1];
        auto src_tile_idx =
            (m * num_k_tiles_ * num_n_tiles_ + n * num_k_tiles_ + k) *
            kernel_z_shape_[0] * kernel_z_shape_[1];

        for (int64_t i = 0; i < kernel_z_shape_[0]; i++) {
          for (int64_t j = 0; j < kernel_z_shape_[1] / 64; j++) {
            auto dest_idx = dest_tile_idx + i * kernel_z_shape_[1] + j * 64;
            auto src_idx = src_tile_idx + i * kernel_z_shape_[1] + j * 64;
            __m512i va = _mm512_load_si512(&c_map_2[dest_idx]);
            __m512i vb = _mm512_load_si512(&c_map_2[src_idx]);
            __m512i vresult = _mm512_add_epi8(va, vb);
            _mm512_store_si512(reinterpret_cast<__m512i *>(&c_map_2[dest_idx]),
                               vresult);
          }
        }
      }
    }
  }
  auto t3 = std::chrono::steady_clock::now();
  auto t_cpu =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  auto t_avx =
      std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
  std::cout << "cpu time: " << t_cpu << "     avx time: " << t_avx << std::endl;
  for (int i = 0; i < C_BO_SIZE_TOKEN; i++) {
    if (c_map[i] != c_map_2[i]) {

      std::cout << "mismatch idx : " << i << "    cpu value : " << (int)c_map[i]
                << "    avx value : " << (int)c_map_2[i] << std::endl;
      return;
    }
  }
}
TEST(avx, INPUT) {
  int SEQ_BYTES = sizeof(ryzenai::GemmSeq);
  std::vector<int> kernel_x_shape = {8, 2048};
  std::vector<int> kernel_y_shape = {2048, 2048};
  std::tuple<int, int> a_shape = {8, 8192};
  std::tuple<int, int> b_shape = {2048, 8192};
  int64_t kernel_z_shape_[2];
  auto M = std::get<0>(a_shape);
  auto K = std::get<1>(a_shape);
  auto N = std::get<1>(b_shape);
  auto num_m_tiles_ = (int64_t)std::ceil((float)M / (float)kernel_x_shape[0]);
  auto num_k_tiles_ = (int64_t)std::ceil((float)K / (float)kernel_y_shape[0]);
  auto num_n_tiles_ = (int64_t)std::ceil((float)N / (float)kernel_y_shape[1]);
  kernel_z_shape_[0] = kernel_x_shape[0];
  kernel_z_shape_[1] = kernel_y_shape[1];
  int64_t C_BO_SIZE_TOKEN = num_m_tiles_ * num_k_tiles_ * kernel_z_shape_[1] *
                            num_n_tiles_ * kernel_z_shape_[0] * sizeof(int8_t);
  kernel_x_shape[0] = 8;
  auto per_kernel_input_size_ =
      kernel_x_shape[0] * kernel_x_shape[1] * sizeof(int8_t) + SEQ_BYTES;
  auto a_size = num_m_tiles_ * num_k_tiles_ * per_kernel_input_size_;
  std::vector<int8_t> a_map;
  a_map.resize(a_size);
  for (int i = 0; i < a_size; i++)
    a_map[i] = i % 64;
  std::vector<int8_t> a;
  auto input_size = std::get<1>(a_shape) * std::get<0>(a_shape);
  a.resize(input_size);
  for (int i = 0; i < a.size(); i++)
    a[i] = i % 64;
  auto a_dtype_size_ = sizeof(int8_t);
  auto act_tile_count = 0;
  std::cout << "memcpy rounds "
            << num_m_tiles_ * num_k_tiles_ * kernel_x_shape[0]
            << "    step size" << per_kernel_input_size_ << std::endl;

  auto t4 = std::chrono::steady_clock::now();

  for (int64_t m = 0; m < num_m_tiles_; m++) {
    for (int64_t k = 0; k < num_k_tiles_; k++) {
      int input_shape[2];
      input_shape[0] =
          std::min((int)(std::get<0>(a_shape) - m * kernel_x_shape[0]),
                   (int)kernel_x_shape[0]);
      input_shape[1] =
          std::min((int)(kernel_x_shape[1]),
                   (int)(std::get<1>(a_shape) - k * kernel_x_shape[1]));
      auto src_idx =
          k * kernel_x_shape[1] + std::get<1>(a_shape) * kernel_x_shape[0] * m;
      auto dest_off = (kernel_x_shape[0] * kernel_x_shape[1] + SEQ_BYTES) *
                      (m * num_k_tiles_ + k) * a_dtype_size_;

      for (int64_t i = 0; i < input_shape[0]; i++) {
        memcpy((void *)((int8_t *)a_map.data() + dest_off +
                        (i * kernel_x_shape[1] * a_dtype_size_)),
               (void *)&a[src_idx + i * std::get<1>(a_shape)],
               input_shape[1] * a_dtype_size_);
      }
    }
  }
  auto t5 = std::chrono::steady_clock::now();
  for (int64_t m = 0; m < num_m_tiles_; m++) {
    for (int64_t k = 0; k < num_k_tiles_; k++) {
      int input_shape[2];
      input_shape[0] =
          std::min((int)(std::get<0>(a_shape) - m * kernel_x_shape[0]),
                   (int)kernel_x_shape[0]);
      input_shape[1] =
          std::min((int)(kernel_x_shape[1]),
                   (int)(std::get<1>(a_shape) - k * kernel_x_shape[1]));
      auto src_idx =
          k * kernel_x_shape[1] + std::get<1>(a_shape) * kernel_x_shape[0] * m;
      auto dest_off = (kernel_x_shape[0] * kernel_x_shape[1] + SEQ_BYTES) *
                      (m * num_k_tiles_ + k) * a_dtype_size_;

      for (int64_t i = 0; i < input_shape[0]; i++) {
        auto d_idx = dest_off + (i * kernel_x_shape[1] * a_dtype_size_);
        if (d_idx % 32) {
          memcpy((void *)((int8_t *)a_map.data() + d_idx),
                 (void *)&a[src_idx + i * std::get<1>(a_shape)],
                 input_shape[1] * a_dtype_size_);
        } else {

          fastMemcpy((void *)((int8_t *)a_map.data() + d_idx),
                     (void *)&a[src_idx + i * std::get<1>(a_shape)],
                     input_shape[1] * a_dtype_size_);
        }
      }
    }
  }
  auto t6 = std::chrono::steady_clock::now();
  auto a_m = Eigen::Map<Eigen::Matrix<int8_t, 8, 8192, Eigen::RowMajor>>(
      &a[0], 8, 8196);

  auto o_m =
      Eigen::Map<Eigen::Matrix<int8_t, 4, 8 * 2048 + 8, Eigen::RowMajor>>(
          &a_map[0], 4, 8 * 2048 + 8);
  auto t7 = std::chrono::steady_clock::now();
  for (int64_t m = 0; m < num_m_tiles_; m++) {
    for (int64_t k = 0; k < num_k_tiles_; k++) {
      int input_shape[2];
      input_shape[0] =
          std::min((int)(std::get<0>(a_shape) - m * kernel_x_shape[0]),
                   (int)kernel_x_shape[0]);
      input_shape[1] =
          std::min((int)(kernel_x_shape[1]),
                   (int)(std::get<1>(a_shape) - k * kernel_x_shape[1]));
      auto src_idx =
          k * kernel_x_shape[1] + std::get<1>(a_shape) * kernel_x_shape[0] * m;
      auto dest_off = (kernel_x_shape[0] * kernel_x_shape[1] + SEQ_BYTES) *
                      (m * num_k_tiles_ + k) * a_dtype_size_;

      o_m.block(m * num_k_tiles_ + k, 0, 1, 8 * 2048) =
          a_m.block(kernel_x_shape[0] * m, k * kernel_x_shape[1],
                    input_shape[0], input_shape[1]);
    }
  }
  auto t8 = std::chrono::steady_clock::now();
  auto i_cpu =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t4).count();
  auto i_avx =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t6 - t5).count();
  auto i_eg =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t8 - t7).count();
  std::cout << "input cpy : " << i_cpu << "     avx: " << i_avx
            << "   eigen : " << i_eg << std::endl;
  std::cout << "test pass" << std::endl;
}
