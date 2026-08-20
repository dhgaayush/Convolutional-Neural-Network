clang++ -std=c++17 -O2 -Iincludes srcs/*.cpp -o cnn \
    $(pkg-config --cflags --libs opencv5)