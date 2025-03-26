# 设置编译器
CXX = g++
NVCC = nvcc

# 编译选项
CXXFLAGS = -std=c++11 -I../
NVCCFLAGS = -O3 -std=c++17 -lcudnn -lcublasLt -lcublas

# 头文件
HEADERS = ../blob.h ../Dense.h ../helper.h ../Layer.h ../matmul.cuh ../Activation.h ../Conv2D.h ../CrossEntropyLoss.h ../mnist.h ../Network.h ../Pooling.h ../Softmax.h

# 目标文件
TARGET = test_net

# 源文件
SRC = test_net.cu

# 目标是可执行文件
all: $(TARGET)

# 编译规则
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET) -I../

# 清理目标
clean:
	rm -f $(TARGET)

# 伪目标
.PHONY: all clean