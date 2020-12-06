#pragma once

#include <algorithm>
#include <iterator>
#include <iostream>
#include <mpi.h>

class Matrix;
class Vector;

/* Matrix-Vectors operators */
Vector operator*(const Matrix& mat, const Vector& vec);

class Vector{
private:
    double *data_;
    int     length_;

public:
    Vector();
    Vector(int n, double d = 0.0);
    Vector(const Vector& vec);
    Vector& operator=(const Vector& vec);
    void Fill(int size, double d);

    double* GetData();
    int size() const;
    ~Vector();

    double& operator[](int i);
    double  operator[](int i) const;

    /* Matrix-Vectors operators */
    friend Vector operator+(const Vector& v1, const Vector& v2);
    friend Vector operator-(const Vector& v1, const Vector& v2);
    friend Vector operator*(double a, const Vector& v);
    friend Vector operator*(const Vector& v, double a);
    friend double DotProduct(const Vector& v1, const Vector& v2);

    friend Vector operator*(const Matrix& mat, const Vector& vec);
};


class Matrix{
private:
    int rows_;
    int cols_;
    double *data_;

public:
    Matrix();
    Matrix(int rows, int cols);
    Matrix(const Matrix& mat);
    Matrix& operator=(const Matrix& mat);

    double* GetData();
    int rows() const;
    int cols() const;

    double& at(int i, int j);
    double  at(int i, int j) const;
    friend std::ostream& operator<<(std::ostream& out, const Matrix& mat);
    ~Matrix();

    /* Matrix-Vectors operators */
    friend Vector operator*(const Matrix& mat, const Vector& vec);
};


void DefineSplittingParameters(
        int     *block_index,
        int     *block_size,
        int      N,
        MPI_Comm comm);

void VectorScatter(
        Vector&  vec_root,
        Vector&  vec_local,
        int     *block_size,
        int     *block_index,
        int      root,
        MPI_Comm comm
        );

void VectorGather(
        Vector&  vec_root,
        Vector&  vec_local,
        int     *block_size,
        int     *block_index,
        int      root,
        MPI_Comm comm
);

void MatrixScatter(
        Matrix&  mat_root,
        Matrix&  mat_local,
        int     *block_size,
        int     *block_index,
        int      root,
        MPI_Comm comm
        );