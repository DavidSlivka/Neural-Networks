#include "../Matrix.h"
#include "../Vector.h"
#include <cassert>
#include <iostream>
#include <typeinfo>


void testMatrixFill() {
    Matrix<float> A(2, 2);
    A.fill(5);
    Matrix<float> expected(2, 2, 5);
    assert(A == expected);
    std::cout << "Matrix Fill Test Passed!" << std::endl;
}

void testMatrixAddBiasColumn() {
    Matrix<float> A(2, 2, 1.0);
    Matrix<float> result = A.addBiasColumn();
    Matrix<float> expected({ {1, 1, 1}, {1, 1, 1} });
    assert(result == expected);
    std::cout << "Matrix Add Bias Column Test Passed!" << std::endl;
}

void testMatrixTranspose() {
    Matrix<float> A({ {1, 2}, {3, 4} });
    Matrix<float> expectedA({ {1, 3}, {2, 4} });
    assert(A.transpose() == expectedA);

    Matrix<float> B({ {1, 2, 3, 4}, {5, 6, 7, 8} });
    Matrix<float> expectedB({ {1, 5}, {2, 6}, {3, 7}, {4, 8} });
    assert(B.transpose() == expectedB);

    std::cout << "Matrix Transpose Test Passed!" << std::endl;
}

void testMatrixScalarOperations() {
    Matrix<float> A({ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} });

    Matrix<float> expected({ {2, 3, 4}, {5, 6, 7}, {8, 9, 10} });
    assert(A + 1 == expected);
    assert(1 + A == expected);

    
    expected = Matrix<float>({ {0, 1, 2}, {3, 4, 5}, {6, 7, 8} });
    assert((A + 1 - 2) == expected);

    expected = Matrix<float>({ {0, 2, 4}, {6, 8, 10}, {12, 14, 16} });
    assert(((A + 1 - 2) * 2) == expected);

    expected = Matrix<float>({ {0, 1, 2}, {3, 4, 5}, {6, 7, 8} });
    assert((((A + 1 - 2) * 2) / 2) == expected);
    
    std::cout << "Matrix Scalar Operations Test Passed!" << std::endl;
}

void testMatrixVectorOperations() {
    Matrix<int> A({ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} });
    Matrix<int> expected({ {2, 3, 5}, {5, 6, 8}, {8, 9, 11} });
    Vector<int> expectedvec({ 9, 21, 33 });
    Vector<int> vec = { 1, 1, 2 };
    assert(A + vec == expected);
    assert(vec + A == expected);

    expected = Matrix<int>({ {0, 1, 1}, {3, 4, 4}, {6, 7, 7} });
    assert(A - vec == expected);

    expected = Matrix<int>({ {0, -1, -1}, {-3, -4, -4}, {-6, -7, -7} });
    assert(vec - A == expected);

    expected = Matrix<int>({ {1, 2, 6}, {4, 5, 12}, {7, 8, 18} });
    assert(A * vec == expectedvec);

    std::cout << "Matrix Vector Operations Test Passed!" << std::endl;
}

void testMatrixOperations() {
    Matrix<double> A({ {1, 2}, {3, 4} });
    Matrix<double> B({ {10, 20}, {30, 40} });
    Matrix<double> expectedAdd({ {11, 22}, {33, 44} });
    assert(A + B == expectedAdd);

    Matrix<double> expectedSub({ {9, 18}, {27, 36} });
    assert(B - A == expectedSub);

    A = Matrix<double>::diagonalOnes(2);
    B = Matrix<double>::random(2, 2);
    assert(A.dot(B) == B);

    Matrix<float> C({ {1, 2}, {3, 4} });
    Matrix<float> D({ {10, 20}, {30, 40} });
    Matrix<float> expectedCD({ {70, 100}, {150, 220} });
    assert(C.dot(D) == expectedCD);

    A = Matrix<double>({ {1, 2}, {3, 4}, {5, 6} });
    B = Matrix<double>({ {10, 20, 30}, {40, 50, 60} });
    Matrix<double> expectedAB({ {90, 120, 150}, {190, 260, 330}, {290, 400, 510} });
    assert(A.dot(B) == expectedAB);

    Matrix<double> expectedBA({ {220, 280}, {490, 640} });
    assert(B.dot(A) == expectedBA);

    std::cout << "Matrix Operations Test Passed!" << std::endl;
}

void testMatrixRank() {
    Matrix<float> A({ {1, 2}, {3, 4} });
    assert(A.rank() == 2);

    A.setValue(1, 0, 0);
    assert(A.rank() == 2);
    A.setValue(1, 1, 0);
    assert(A.rank() == 1);

    Matrix<float> B({ {1, 2, 3, 4}, {5, 6, 7, 8} });
    assert(B.rank() == 2);
    assert(B.transpose().rank() == 2);

    B.setValue(1, 0, 0);
    B.setValue(1, 1, 0);
    assert(B.rank() == 2);
    B.setValue(1, 2, 0);
    B.setValue(1, 3, 0);
    assert(B.rank() == 1);

    Matrix<float> D = Matrix<float>::diagonalOnes(5);
    assert(D.rank() == 5);

    std::cout << "Matrix Rank Test Passed!" << std::endl;
}

void testMatrixInverse() {
    Matrix<double> A({ {4, 7}, {2, 6} });
    Matrix<double> Ainv({ {0.6, -0.7}, {-0.2, 0.4} });

    assert(A.inverse().isEqual(Ainv));
    assert(A.dot(Ainv) == Matrix<double>::diagonalOnes(2));

    Matrix<int> B({ {2, 1}, {7, 4} });
    auto Binv = B.inverse(); 
    Matrix<double> expectedBinv({ {4, -1}, {-7, 2} });
    assert(Binv == expectedBinv);

    Matrix<int> C({ {2, 3}, {1, 4} });
    auto Cinv = C.inverse(); 
    Matrix<double> expectedCinv({ {0.8, -0.6}, {-0.2, 0.4} });
    assert(Cinv == expectedCinv);

    std::cout << "Matrix Inverse Test Passed!" << std::endl;
}

int testMatrix() {
    testMatrixFill();
    testMatrixAddBiasColumn();
    testMatrixTranspose();
    testMatrixScalarOperations();
    testMatrixVectorOperations();
    testMatrixOperations();
    testMatrixRank();
    testMatrixInverse();

    std::cout << "All Tests Passed!" << std::endl;
    return 0;
}