#include <iostream>

int main(){

    float a = 0.479998;
    std::cout<<a<<std::endl;
    int b = (int)(a *10 + 0.5);
    std::cout<<b<<std::endl;

    double c = (double)b/10;

    std::cout<<c<<std::endl;


    return 0;
}