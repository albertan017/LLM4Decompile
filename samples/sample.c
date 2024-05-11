#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int func0(float numbers[], int size, float threshold) {
    int i, j;

    for (i = 0; i < size; i++)
        for (j = i + 1; j < size; j++)
            if (fabs(numbers[i] - numbers[j]) < threshold)
                return 1;

    return 0;
}
#include <stdio.h>
#include <assert.h>

int main(){
    float a[] = {1.0, 2.0, 3.9, 4.0, 5.0, 2.2};
    assert(func0(a, 6, 0.3) == 1);
    assert(func0(a, 6, 0.05) == 0);

    float b[] = {1.0, 2.0, 5.9, 4.0, 5.0};
    assert(func0(b, 5, 0.95) == 1);
    assert(func0(b, 5, 0.8) == 0);
    
    float c[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    assert(func0(c, 5, 2.0) == 1);

    float d[] = {1.1, 2.2, 3.1, 4.1, 5.1};
    assert(func0(d, 5, 1.0) == 1);
    assert(func0(d, 5, 0.5) == 0);

    return 0;
}
