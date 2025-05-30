#include "../Vector.h"
#include <cassert>
#include <iostream>
#include <typeinfo>
#include <type_traits>

#include "../Matrix.h"


void testVectorInit() {
	Vector<int> vec(5, 1);
	assert(vec.size() == 5);
	Vector<int> expected({ 1,1,1,1,1 });
	assert(vec == expected);

	std::cout << "Test Init Passed" << std::endl;
}

void testVectorAt() {
	Vector<int> vec({1, 2, 3, 4, 5});
	assert(vec.at(0) == 1);
	assert(vec.at(1) == 2);
	assert(vec.at(2) == 3);
	assert(vec.at(3) == 4);
	assert(vec.at(4) == 5);	
	
	assert(vec[0] == 1);
	assert(vec[1] == 2);
	assert(vec[2] == 3);
	assert(vec[3] == 4);
	assert(vec[4] == 5);


	std::cout << "Test At and [] Passed" << std::endl;
}

void testVectorScalarOperations() {
	Vector<int> vec({ 1, 2, 3, 4, 5 });
	Vector<float> expected({ 1.5, 3.0, 4.5, 6.0, 7.5 });
	float scalar = 1.5;
	assert(vec * scalar == expected);
	assert(scalar * vec == expected);

	expected = Vector<float>({ 2.5, 3.5, 4.5, 5.5, 6.5 });
	assert(vec + scalar == expected);
	assert(scalar + vec == expected);

	expected = Vector<float>({ -0.5, 0.5, 1.5, 2.5, 3.5 });
	assert(vec - scalar == expected);

	expected = Vector<float>({ 0.5, -0.5, -1.5, -2.5, -3.5 });
	assert(scalar - vec == expected);

	vec = Vector<double>({5.0, 10.0, 15.0});
	expected = Vector<float>({2.5, 5.0, 7.5});
	scalar = 2.0;
	assert(vec / scalar == expected);

	std::cout << "Test Scalar Operations Passed" << std::endl;
}

void testVectorTranspose() {
	Vector<int> vec({ 1, 2, 3, 4, 5 });
	vec.transpose();
	auto val = vec.orientation();
	assert(val == Orientation::Row);
	assert(vec.size() == 5);

	std::cout << "Test Transpose Passed" << std::endl;
}

void testVectorOperations() {
	Vector<int> vec1({ 1, 2, 3, 4, 5 });
	Vector<float> vec2({ 1.5, 3.0, 4.5, 6.0, 7.5 });
	Vector<float> expected({ 2.5, 5.0, 7.5, 10.0, 12.5 });
	assert(vec1 + vec2 == expected);
	assert(vec2 + vec1 == expected);

	expected = Vector<float>({0.5, 1.0, 1.5, 2.0, 2.5});
	assert(vec2 - vec1 == expected);

	expected = Vector<float>({ -0.5, -1.0, -1.5, -2.0, -2.5 });
	assert(vec1 - vec2 == expected);

	vec1 = Vector<int>({ 1,1,2 }, Orientation::Row);
	vec2 = Vector<int>({ 1,1,3 }, Orientation::Column);

	int expectedSum = 8;
	assert(vec1.dot(vec2) == expectedSum);

	Matrix<int> ExpectedMatrix = Matrix<int>({ {1, 1, 2}, {1, 1, 2}, {3, 3, 6} });
	assert(vec2.outer(vec1) == ExpectedMatrix);

	std::cout << "Test Vector Operations Passed" << std::endl;
}

int testVector() {
	testVectorInit();
	testVectorAt();
	testVectorScalarOperations();
	testVectorTranspose();
	testVectorOperations();

	return 0;
}
