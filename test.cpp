// dma -> intiailizing 2d arrays of floats and strings
#include <iostream>
#include <string>

int main() {
    int rows = 3;
    int columns = 4;

    // dynamically allocate a two dimensional array of floating values
    float ** dynamicFloatArray = new float * [rows];
    for (int i = 0; i < rows;i++) {
        dynamicFloatArray[i] = new float[columns];
    }


    return 0;


}

