#include "../Perceptron.h"

void testInitPerceptron() {
    Perceptron p(2);
    assert(p.predict({ 1, 1 }) == 0);

    std::cout << "Test Init Perceptron Passed!" << std::endl;
}


void testAndPerceptron() {
    Perceptron p(2, 0.1);
    Vector<Vector<double>> data({
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    });
    Vector<int> labels({ 0, 0, 0, 1 });

    p.train(data, labels, 10);

    assert(p.predict({ 0, 0 }) == 0);
    assert(p.predict({ 0, 1 }) == 0);
    assert(p.predict({ 1, 0 }) == 0);
    assert(p.predict({ 1, 1 }) == 1);

    std::cout << "Test AND Perceptron Passed!" << std::endl;
}


void testOrPerceptron() {
    Perceptron p(2, 0.1);
    Vector<Vector<double>> data({
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    });
    Vector<int> labels = { 0, 1, 1, 1 };

    p.train(data, labels, 10);

    assert(p.predict({ 0, 0 }) == 0);
    assert(p.predict({ 0, 1 }) == 1);
    assert(p.predict({ 1, 0 }) == 1);
    assert(p.predict({ 1, 1 }) == 1);
    
    std::cout << "Test OR Perceptron Passed!" << std::endl;
}


void testZeroInput() {
    Perceptron p(3, 0.1);
    Vector<Vector<double>> data = {
        {0, 0, 0}, {0, 0, 0}
    };
    Vector<int> labels = { 0, 0 };

    p.train(data, labels, 10);

    assert(p.predict({ 0, 0, 0 }) == 0);

    std::cout << "Test Zero input test passed!" << std::endl;
}



int testPerceptron() {
    testInitPerceptron();
    testAndPerceptron();
    testOrPerceptron();
    testZeroInput();

    return 0;
}