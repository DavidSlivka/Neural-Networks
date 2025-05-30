#ifndef TEST_DATA_STORAGE_H
#define TEST_DATA_STORAGE_H

#include "../DataStorage.h"
#include <iostream>
#include <cassert>

void test_read_csv_and_print() {
    DataStorage df;
    df.readCsv("test.csv", SEMICOLON);
    df.print();
    assert(!df.columns().empty());
}

void test_head_tail_shape() {
    DataStorage df;
    df.readCsv("test.csv", SEMICOLON);

    std::cout << "Head:" << NEW_LINE;
    df.head(2);
    std::cout << "Tail:" << NEW_LINE;
    df.tail(2);

    std::pair<int, int> size = df.shape();
    assert(size.first > 0 && size.second > 0);
}

void test_columns_dtypes() {
    DataStorage df;
    df.readCsv("test.csv", SEMICOLON);
    auto cols = df.columns();
    auto types = df.dtypes();

    assert(cols.size() == types.size());
    for (const auto& dtype : types)
        assert(dtype == "float");
}

void test_describe() {
    DataStorage df;
    df.readCsv("test.csv", SEMICOLON);
    df.describe();
}

void test_dropna_fillna() {
    DataStorage df;
    df.readCsv("test_with_missing.csv", SEMICOLON);
    df.print();
    df.fillna(0.0f);
    auto corr = df.correlationMatrix();
    auto cov = df.covarianceMatrix();

    corr.print();
    std::cout << std::endl;
    cov.print();
    df.dropna();

    DataStorage df2;
    df2.readCsv("test_with_missing.csv", SEMICOLON);
    df2.print();
    df2.fillna(0.0f);
    df2.print();
}

void test_drop_rename() {
    DataStorage df;
    df.readCsv("test.csv", SEMICOLON);

    auto original_cols = df.columns();
    df.drop(original_cols[0]);
    assert(df.columns().size() == original_cols.size() - 1);

    df.rename(df.columns()[0], "new_name");
    assert(df.columns()[0] == "new_name");
}

void test_correlation_covariance() {
    DataStorage df;
    df.readCsv("test.csv", SEMICOLON);

    auto corr = df.correlationMatrix();
    auto cov = df.covarianceMatrix();

    corr.print();
    std::cout << std::endl;
    cov.print();

    assert(corr.isSquare());
    assert(cov.isSquare());
}

void test_train_test_split() {
    DataStorage df;
    df.readCsv("test.csv", SEMICOLON);

    float split_point = 0.6f;

    std::pair<DataStorage, DataStorage> out = df.trainTestSplit(split_point);
    assert(out.first.shape().first == (int)(df.shape().first * split_point));
    assert(out.first.shape().second== (int)df.shape().second);
    assert(out.second.shape().first == (int)(df.shape().first * 0.4));
    assert(out.second.shape().second== (int)df.shape().second);
}

void test_oneHotEncodeColumn() {
    DataStorage df;

    df.readCsv("test_onehot.csv", SEMICOLON);

    df.oneHotEncodeColumn(1);

    assert(df.getHeaders().size() == 4);
    assert(df.getHeaders()[0] == "feature");
    assert(df.getHeaders()[1] == "onehot_0");
    assert(df.getHeaders()[2] == "onehot_1");
    assert(df.getHeaders()[3] == "onehot_2");

    assert(df.getData()[0][0] == 10);
    assert(df.getData()[0][1] == 20);
    assert(df.getData()[0][2] == 30);
    assert(df.getData()[0][3] == 40);

    assert(df.getData()[1][0] == 1);
    assert(df.getData()[2][0] == 0);
    assert(df.getData()[3][0] == 0);

    assert(df.getData()[1][1] == 0);
    assert(df.getData()[2][1] == 1);
    assert(df.getData()[3][1] == 0);

    assert(df.getData()[1][2] == 1);
    assert(df.getData()[2][2] == 0);
    assert(df.getData()[3][2] == 0);

    assert(df.getData()[1][3] == 0);
    assert(df.getData()[2][3] == 0);
    assert(df.getData()[3][3] == 1);
}

void test_normalizeMinMax() {
    DataStorage df;

    df.readCsv("test_normalization.csv", SEMICOLON);

    df.normalizeMinMax(0);

    // original min = 10, max = 50
    // scaled value = (x - 10) / (50 - 10)
    std::vector<float> expected = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };

    for (size_t i = 0; i < df.getData()[0].size(); ++i) {
        assert(std::abs(df.getData()[0][i] - expected[i]) < EPSILON);
    }

    std::cout << "test_normalizeMinMax passed!" << NEW_LINE;
}

void test_normalizeZScore() {
    DataStorage df;

    df.readCsv("test_normalization.csv", SEMICOLON);

    df.normalizeZScore(0);

    // mean = 30, stddev = sqrt(200) ≈ 14.1421
    // (x - mean) / stddev
    std::vector<float> expected = {
        (10 - 30) / std::sqrt(200.0f),
        (20 - 30) / std::sqrt(200.0f),
        (30 - 30) / std::sqrt(200.0f),
        (40 - 30) / std::sqrt(200.0f),
        (50 - 30) / std::sqrt(200.0f)
    };

    for (size_t i = 0; i < df.getData()[0].size(); ++i) {
        assert(std::abs(df.getData()[0][i] - expected[i]) < EPSILON);
    }

    std::cout << "test_normalizeZScore passed!" << NEW_LINE;
}



int testDataStorageClass() {
    test_read_csv_and_print();
    test_head_tail_shape();
    test_columns_dtypes();
    test_describe();
    test_dropna_fillna();
    test_drop_rename();
    test_correlation_covariance();
    test_train_test_split();
    test_oneHotEncodeColumn();
    test_normalizeMinMax();
    test_normalizeZScore();

    std::cout << "All tests passed!\n";
    return 0;
}

#endif