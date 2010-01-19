#include "activate.h"

int main()
{
    Activate stim(argv[1]);
    
    for(int i = 0 ; i < 1000; i ++) {
        cout << i/10. << " " << stim.at(i/10.) << endl;
    }

}
