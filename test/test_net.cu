#include "../Network.h"
#include "../mnist.h"
#include "../helper.h"

#include <iomanip>
#include <nvtx3/nvToolsExt.h>
#include <chrono>

int main()
{
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    /* configure the network */
    int batch_size_train = 512;
    int num_steps_train = 2400;
    int monitoring_step = 200;

    float initial_learning_rate = 0.02f;
    float learning_rate = 0.0;
    float lr_decay = 0.0005f;

    bool load_pretrain = false;
    bool file_save = false;

    int batch_size_test = 10;
    int num_steps_test = 1000;

    /* Welcome Message */
    std::cout << "== MNIST training with CUDNN ==" << std::endl;

    // phase 1. training
    std::cout << "[TRAIN]" << std::endl;

    /* Welcome Message */
    std::cout << "== MNIST training with CUDNN ==" << std::endl;

    // phase 1. training
    std::cout << "[TRAIN]" << std::endl;

    using ftype = __half;

    // step 1. loading dataset
    MNIST<ftype>
        train_data_loader = MNIST<ftype>("../dataset");
    train_data_loader.train(batch_size_train);

    // step 2. model initialization
    Network<ftype> model;
    model.add_layer(new Conv2D<ftype>("conv1", 20, 5));
    model.add_layer(new Pooling<ftype>("pool", 2, 0, 2, CUDNN_POOLING_MAX));
    model.add_layer(new Conv2D<ftype>("conv2", 50, 5));
    model.add_layer(new Pooling<ftype>("pool", 2, 0, 2, CUDNN_POOLING_MAX));
    model.add_layer(new Dense<ftype>("dense1", 500));
    model.add_layer(new Activation<ftype>("relu", CUDNN_ACTIVATION_RELU));
    model.add_layer(new Dense<ftype>("dense2", 10));
    model.add_layer(new Softmax<ftype>("softmax"));
    model.cuda();

    if (load_pretrain)
        model.load_pretrain();
    model.train();

    // train
    int step = 1;
    Blob<ftype> *train_data = train_data_loader.get_data();
    Blob<ftype> *train_target = train_data_loader.get_target();
    train_data_loader.get_batch();

    while (step < num_steps_train)
    {

        // update shared buffer contents
        train_data->to(cuda);
        train_target->to(cuda);

        // forward
        Blob<ftype> *out_pr = model.forward(train_data);
        // backward
        model.backward(train_target);
        learning_rate = initial_learning_rate / (1.f + lr_decay * step);

        model.update(learning_rate, step);

        if (step % monitoring_step == 0)
        {
            float loss = model.loss(out_pr, train_target);
            float accuracy = model.get_accuracy(out_pr, train_target);

            std::cout << "step: " << std::right << std::setw(4) << step << ", loss: " << std::left << std::setw(5) << std::fixed << std::setprecision(3) << loss << ", accuracy: " << accuracy << "%" << std::endl;
        }
        step = train_data_loader.next();
    }

    // test
    // step 1. load test set
    std::cout << "[INFERENCE]" << std::endl;
    MNIST<ftype> test_data_loader = MNIST<ftype>("../dataset");
    test_data_loader.test(batch_size_test);

    // step 2. model initialization
    model.test();

    // step 3. iterates the testing loop
    Blob<ftype> *test_data = test_data_loader.get_data();
    Blob<ftype> *test_target = test_data_loader.get_target();
    test_data_loader.get_batch();
    float acc = 0.0f;
    step = 0;
    Blob<ftype> *out_pr;

    auto start = std::chrono::high_resolution_clock::now(); // 记录起始时间

    while (step < num_steps_test)
    {

        // update shared buffer contents
        test_data->to(cuda);
        test_target->to(cuda);

        // forward
        out_pr = model.forward(test_data);
        acc += model.get_accuracy(out_pr, test_target);

        // fetch next data
        step = test_data_loader.next();
    }
    auto end = std::chrono::high_resolution_clock::now(); // 记录结束时间
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "执行时间: " << elapsed.count() << " 毫秒" << std::endl;

    // step 4. calculate loss and accuracy
    float loss = model.loss(out_pr, train_target);

    std::cout << "loss: " << std::setw(4) << loss << ", accuracy: " << acc / num_steps_test << "%" << std::endl;

    // Good bye
    std::cout << "Done." << std::endl;

    return 0;
}
