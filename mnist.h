#pragma once

#include <string>
#include <fstream>
#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <assert.h>
#include <cstring>
#include <iomanip>
#include "blob.h"
#include <type_traits>

#define MNIST_CLASS 10
template <typename ftype>
class MNIST
{
public:
    MNIST() : dataset_dir_("./") {}
    MNIST(std::string dataset_dir) : dataset_dir_(dataset_dir) {}
    ~MNIST();

    // load train dataset
    void train(int batch_size = 1, bool shuffle = false);

    // load test dataset
    void test(int batch_size = 1);

    // update shared batch data buffer at current step index
    void get_batch();
    // increase current step index
    // optionally it updates shared buffer if input parameter is true.
    int next();

    // returns a pointer which has input batch data
    Blob<ftype> *get_data() { return data_; }
    // returns a pointer which has target batch data
    Blob<ftype> *get_target() { return target_; }

private:
    // predefined file names
    std::string dataset_dir_; // 数据存放地址

    std::string train_dataset_file_ = "train-images-idx3-ubyte"; // 训练数据集
    std::string train_label_file_ = "train-labels-idx1-ubyte";   // 训练标签集
    std::string test_dataset_file_ = "t10k-images-idx3-ubyte";   // 测试数据集
    std::string test_label_file_ = "t10k-labels-idx1-ubyte";     // 测试标签集

    // container
    std::vector<std::vector<ftype>> data_pool_;
    std::vector<std::array<ftype, MNIST_CLASS>> target_pool_;
    Blob<ftype> *data_ = nullptr;   // 图像信息
    Blob<ftype> *target_ = nullptr; // 标签就是答案

    // data loader initialization
    void load_data(std::string &image_file_path);
    void load_target(std::string &label_file_path);

    void normalize_data();

    int to_int(uint8_t *ptr);

    // data loader control
    int step_ = -1;
    bool shuffle_;
    int batch_size_ = 1;
    int channels_ = 1;
    int height_ = 1;
    int width_ = 1;
    int num_classes_ = 10; // 一共有十种数字类型
    int num_steps_ = 0;

    void create_shared_space();
    void shuffle_dataset();
};
template <typename ftype>
MNIST<ftype>::~MNIST()
{
    delete data_;
    delete target_;
}
template <typename ftype>
void MNIST<ftype>::create_shared_space()
{
    // create blobs with batch size and sample size
    data_ = new Blob<ftype>(batch_size_, channels_, height_, width_);
    data_->tensor();
    target_ = new Blob<ftype>(batch_size_, num_classes_);
}
template <typename ftype>
void MNIST<ftype>::load_data(std::string &image_file_path)
{
    uint8_t ptr[4];
    std::string file_path_ = dataset_dir_ + "/" + image_file_path;

    std::cout << "loading " << file_path_ << std::endl;
    std::ifstream file(file_path_.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        std::cout << "Download dataset first!!" << std::endl;
        std::cout << "You can get the MNIST dataset from 'http://yann.lecun.com/exdb/mnist/' or just use 'download_mnist.sh' file." << std::endl;
        exit(-1);
    }

    file.read((char *)ptr, 4);
    int magic_number = to_int(ptr);
    assert((magic_number & 0xFFF) == 0x803);

    int num_data;
    file.read((char *)ptr, 4);
    num_data = to_int(ptr); // 60000
    file.read((char *)ptr, 4);
    height_ = to_int(ptr); // 28
    file.read((char *)ptr, 4);
    width_ = to_int(ptr); // 28

    std::cout << num_data << " " << height_ << " " << width_ << std::endl;
    uint8_t *q = new uint8_t[channels_ * height_ * width_];
    for (int i = 0; i < num_data; i++)
    {
        std::vector<ftype> image = std::vector<ftype>(channels_ * height_ * width_);
        ftype *image_ptr = image.data();

        file.read((char *)q, channels_ * height_ * width_);
        for (int j = 0; j < channels_ * height_ * width_; j++)
        {
            if constexpr (std::is_same_v<ftype, float>)
            {
                image_ptr[j] = static_cast<float>(q[j]) / 255.f;
            }
            else if constexpr (std::is_same_v<ftype, __half>)
            { // FP16 (half)
                image_ptr[j] = __float2half(static_cast<float>(q[j]) / 255.f);
            }
            else if constexpr (std::is_same_v<ftype, __nv_bfloat16>)
            { // BF16
                image_ptr[j] = __float2bfloat16(static_cast<float>(q[j]) / 255.f);
            }
            else
            {
                static_assert(sizeof(ftype) == 0, "Unsupported type for ftype");
            }
        }

        data_pool_.push_back(image);
    }

    delete[] q;

    num_steps_ = num_data / batch_size_;

    std::cout << "loaded " << data_pool_.size() << " items.." << std::endl;
    //  for (auto it : data_pool_)
    //  {
    //      for (int i = 0; i < 28; i++)
    //      {
    //          for (int j = 0; j < 28; j++)
    //          {
    //              std::cout << std::fixed << std::setprecision(1) << std::setw(2) << (float)it[i * 28 + j] << " " << sizeof(it[i * 28 + j]) << " ";
    //          }
    //          std::cout << std::endl;
    //      }
    //      break;
    //  }
    file.close();
}
template <typename ftype>
void MNIST<ftype>::load_target(std::string &label_file_path) // 读取特征
{
    uint8_t ptr[4];
    std::string file_path_ = dataset_dir_ + "/" + label_file_path;

    std::ifstream file(file_path_.c_str(), std::ios::in | std::ios::binary);

    if (!file.is_open())
    {
        std::cout << "Check dataset existance!!" << std::endl;
        exit(-1);
    }

    file.read((char *)ptr, 4);
    int magic_number = to_int(ptr);
    assert((magic_number & 0xFFF) == 0x801);

    file.read((char *)ptr, 4);
    int num_target = to_int(ptr);
    std::cout << num_target << std::endl;
    // prepare input buffer for label
    // read all labels and converts to one-hot encoding
    for (int i = 0; i < num_target; i++)
    {
        std::array<ftype, MNIST_CLASS> target_batch;
        if constexpr (std::is_same_v<ftype, float>)
        {
            std::fill(target_batch.begin(), target_batch.end(), static_cast<float>(0.0f));
        }
        else if constexpr (std::is_same_v<ftype, __half>)
        { // FP16 (half)
            std::fill(target_batch.begin(), target_batch.end(), __float2half(0.0f));
        }
        else if constexpr (std::is_same_v<ftype, __nv_bfloat16>)
        { // BF16
            std::fill(target_batch.begin(), target_batch.end(), __float2bfloat16(0.0f));
        }
        // std::fill(target_batch.begin(), target_batch.end(), ftype(0.0f));

        file.read((char *)ptr, 1);
        // target_batch[static_cast<int>(ptr[0])] = ftype(1.f);
        if constexpr (std::is_same_v<ftype, float>)
        {
            target_batch[static_cast<int>(ptr[0])] = static_cast<float>(1.f);
        }
        else if constexpr (std::is_same_v<ftype, __half>)
        { // FP16 (half)
            target_batch[static_cast<int>(ptr[0])] = __float2half(1.f);
        }
        else if constexpr (std::is_same_v<ftype, __nv_bfloat16>)
        { // BF16
            target_batch[static_cast<int>(ptr[0])] = __float2bfloat16(1.f);
        }

        target_pool_.push_back(target_batch);
    }

    file.close();
}
template <typename ftype>
void MNIST<ftype>::shuffle_dataset() // 随机打乱数据集
{
    std::random_device rd;
    std::mt19937 g_data(rd());
    auto g_target = g_data;

    std::shuffle(std::begin(data_pool_), std::end(data_pool_), g_data);
    std::shuffle(std::begin(target_pool_), std::end(target_pool_), g_target);
}
template <typename ftype>
int MNIST<ftype>::to_int(uint8_t *ptr)
{
    return ((ptr[0] & 0xFF) << 24 | (ptr[1] & 0xFF) << 16 |
            (ptr[2] & 0xFF) << 8 | (ptr[3] & 0xFF) << 0);
}
template <typename ftype>
void MNIST<ftype>::train(int batch_size, bool shuffle)
{
    if (batch_size < 1)
    {
        std::cout << "batch size should be greater than 1." << std::endl;
        return;
    }

    batch_size_ = batch_size;
    shuffle_ = shuffle;

    load_data(train_dataset_file_);
    load_target(train_label_file_);

    if (shuffle_)
        shuffle_dataset(); // 随机打乱
    create_shared_space();

    step_ = 0;
}
template <typename ftype>
void MNIST<ftype>::test(int batch_size)
{
    if (batch_size < 1)
    {
        std::cout << "batch size should be greater than or equal to 1." << std::endl;
        return;
    }

    batch_size_ = batch_size;

    load_data(test_dataset_file_);
    load_target(test_label_file_);

    create_shared_space();

    step_ = 0;
}
template <typename ftype>
void MNIST<ftype>::get_batch()
{
    if (step_ < 0)
    {
        std::cout << "You must initialize dataset first.." << std::endl;
        exit(-1);
    }

    // index clipping
    int data_idx = step_ % num_steps_ * batch_size_;

    // prepare data blob
    int data_size = channels_ * width_ * height_;

    // copy data
    for (int i = 0; i < batch_size_; i++) // 取出来一个batch_size
        std::copy(data_pool_[data_idx + i].data(),
                  &data_pool_[data_idx + i].data()[data_size],
                  &data_->ptr()[data_size * i]);
    // copy target with one-hot encoded
    for (int i = 0; i < batch_size_; i++)
        std::copy(target_pool_[data_idx + i].data(),
                  &target_pool_[data_idx + i].data()[MNIST_CLASS],
                  &target_->ptr()[MNIST_CLASS * i]);
}
template <typename ftype>
int MNIST<ftype>::next()
{
    step_++;

    get_batch();

    return step_;
}
