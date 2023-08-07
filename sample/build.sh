hipcc -c -o saxpy_hip.o saxpy_hip.cpp
clang++ -lHIPLoader -o saxpy_hip saxpy_hip.o
