#include <stdlib.h>
#include <stdio.h>

int main()
{
    float a = 65.6666;
    float b = 65.999;
    unsigned char achar = (unsigned char) a;
    unsigned char bchar = (unsigned char) b;
    printf("%d %d",achar, bchar);
    return 1;
}