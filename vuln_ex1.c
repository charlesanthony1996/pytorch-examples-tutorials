#include <stdio.h>
#include <string.h>

void vulnerable_function(char *str) {
    char buffer[100];
    strcpy(buffer, str);
}

int main(int argc, char **argv) {
    if(argc < 2) {
        printf("");
    }

}