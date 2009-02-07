#include "particle.h"
#include <itkVector.h>

class test : PBase<1> {
public:
    test();
    itk::Vector<double, 1> step(itk::Vector<double, 1>);
    double error(itk::Vector<double, 1>);
    double input;
};

test::test()
{

}

itk::Vector<double, 1> test::step(itk::Vector<double, 1> state)
{
    return state;
}

double test::error(itk::Vector<double, 1> state)
{
    return state[0] - input < 0 ? input - state[0] : state[0] - input;
}

int main()
{
    test test1;
    ParticleF<test, 1> pfilter;
    
    itk::Statistics::NormalVariateGenerator rv;
    rv.Initialize(rand());
    double obs;
    for(int i = 0 ; i<1000 ; i++) {
        if(rand() % 2 ) {
            obs = rv.GetVariate() - 10;
        } else {
            obs = rv.GetVariate() + 10;
        }
        test.input = obs;
        pfilter.update(test1, 1);
    }
}
