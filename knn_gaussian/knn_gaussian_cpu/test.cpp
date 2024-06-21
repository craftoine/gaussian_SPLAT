#include <iostream>
template<bool B>
class bidon {
public:
    #if B
       int a[10];
    #endif
    int b[5*(((int)B/)+1)];
    bidon(){
        #if B
        std::cout << "B is true" << std::endl;
        #else
        std::cout << "B is false" << std::endl;
        #endif
    }
};
int main(){
    bidon<true> b1;
    bidon<false> b2;
    return 0;
}