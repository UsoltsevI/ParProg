#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int compare(const char* filename1, const char* filename2, int K, int* last_el) {
    FILE *f1 = fopen(filename1, "r");
    FILE *f2 = fopen(filename2, "r");
    
    if (!f1 || !f2){
        fprintf(stderr, "Failed to open file");
        return 0;
    }

    int a, b;
    int last = 0;

    while (last < K && ((a = fgetc(f1)) != EOF) && ((b = fgetc(f2)) != EOF)) {
        if (a != b) {
            *last_el = last; 
            fclose(f1);
            fclose(f2);
            return a - b;
        }
        last++;
    }

    *last_el = last;
    fclose(f1);
    fclose(f2);
    return 0;

}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <file_name_1> <file_name_2> <K>\n", argv[0]);
    }
    int last_el;
    int result = compare(argv[1], argv[2], atoi(argv[3]), &last_el);
    printf("Result: %d, last_el = %d\n", result, last_el);
    return 0;
}