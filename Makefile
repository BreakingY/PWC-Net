# Makefile for infer_trt
CXX := g++
CXXFLAGS := -g -std=c++17 \
            -I/data/sunkx/TensorRT-10.4.0.26/include \
            -I/usr/local/cuda/include \
            -I/usr/local/include/opencv4


LDFLAGS := -L/data/sunkx/TensorRT-10.4.0.26/lib \
           -L/usr/local/cuda/lib64 \
           -L/usr/local/lib


LDLIBS := -lnvinfer -lcudart \
          -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio \
		  -lnppig -lnppidei -lnppial


TARGET := infer_trt
SRC := infer_trt.cpp


all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS) $(LDLIBS)


clean:
	rm -f $(TARGET)
