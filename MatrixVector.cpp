#include "MatrixVector.h"

Vector::Vector() :
        data_(nullptr),
        length_(0){

}

Vector::Vector(int n, double d) :
        length_(n)
{
    data_ = new double[length_];
    for (int i = 0; i < length_; ++i){
        data_[i] = d;
    }
}

Vector::Vector(const Vector& vec){
    length_ = vec.length_;
    data_ = new double[length_];
    std::copy(vec.data_, vec.data_ + vec.length_, data_);
}

Vector& Vector::operator=(const Vector &vec)
{
    if (this == &vec){
        return *this;
    }
    if (data_ != nullptr){
        delete[] data_;
    }
    length_ = vec.length_;
    data_ = new double[length_];
    std::copy(vec.data_, vec.data_ + vec.length_, data_);
    return *this;
}

void Vector::Fill(int size, double d)
{
    if (data_ != nullptr){
        delete[] data_;
    }
    data_ = new double[size];
    length_ = size;
    for (int i = 0; i < length_; ++i){
        data_[i] = d;
    }
}

double* Vector::GetData(){
    return data_;
}

int Vector::size() const{
    return length_;
}

Vector::~Vector(){
    delete[] data_;
}

double& Vector::operator[](int i)
{
    return data_[i];
}

double Vector::operator[](int i) const
{
    return data_[i];
}


Matrix::Matrix() :
        rows_(0),
        cols_(0),
        data_(nullptr){

}

Matrix::Matrix(int rows, int cols)
{
    rows_ = rows;
    cols_ = cols;
    data_ = new double[rows_ * cols_];
}

Matrix::Matrix(const Matrix &mat)
{
    rows_ = mat.rows_;
    cols_ = mat.cols_;
    data_ = new double[rows_ * cols_];
    std::copy(mat.data_, mat.data_ + rows_ * cols_, data_);
}

Matrix& Matrix::operator=(const Matrix &mat)
{
    if (this == &mat){
        return *this;
    }
    if (data_ != nullptr){
        delete[] data_;
    }
    rows_ = mat.rows_;
    cols_ = mat.cols_;
    data_ = new double[rows_ * cols_];
    std::copy(mat.data_, mat.data_ + rows_ * cols_, data_);
    return *this;
}

double* Matrix::GetData()
{
    return data_;
}

int Matrix::rows() const
{
    return rows_;
}

int Matrix::cols() const
{
    return cols_;
}

double& Matrix::at(int i, int j)
{
    return *(data_ + i*cols_ + j);
}

double Matrix::at(int i, int j) const
{
    return *(data_ + i*cols_ + j);
}

std::ostream& operator<<(std::ostream& out, const Matrix& mat){
    for (int i = 0; i < mat.rows(); ++i){
        for (int j = 0; j < mat.cols() - 1; ++j){
            out << mat.at(i, j) << " ";
        }
        out << mat.at(i, mat.cols() - 1) << std::endl;
    }
    return out;
}

Matrix::~Matrix()
{
    delete[] data_;
}




/* Matrix-Vectors operators */
Vector operator+(const Vector& v1, const Vector& v2){
    int size = v1.size();
    Vector result(size);
    for (int i = 0; i < size; ++i){
        result[i] = v1[i] + v2[i];
    }
    return result;
}

Vector operator-(const Vector& v1, const Vector& v2){
    int size = v1.size();
    Vector result(size);
    for (int i = 0; i < size; ++i){
        result[i] = v1[i] - v2[i];
    }
    return result;
}

Vector operator*(double a, const Vector& v){
    int size = v.size();
    Vector result = v;
    for (int i = 0; i < size; ++i){
        result[i] *= a;
    }
    return result;
}

Vector operator*(const Vector& v, double a){
    int size = v.size();
    Vector result = v;
    for (int i = 0; i < size; ++i){
        result[i] *= a;
    }
    return result;
}

double DotProduct(const Vector& v1, const Vector& v2){
    double result = 0.0;
    for (int i = 0; i < v1.size(); ++i){
        result += v1[i] * v2[i];
    }
    return result;
}

Vector operator*(const Matrix& mat, const Vector& vec){
    Vector result(vec.size());
    for (int i = 0; i < mat.rows(); ++i){
        result[i] = 0.0;
        for (int j = 0; j < mat.cols(); ++j){
            result[i] += mat.at(i, j) * vec[j];
        }
    }
    return result;
}


void DefineSplittingParameters(
        int     *block_index,
        int     *block_size,
        int      N,
        MPI_Comm comm)
{
    int p;
    int id;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &id);

    int N_local   = N / p;
    int index     = id * N_local;
    int row_shift = N % p;

    if (id < N % p){
        N_local += 1;
        row_shift = id;
    }
    index += row_shift;

    MPI_Allgather(
            &N_local,
            1,
            MPI_INT,
            block_size,
            1,
            MPI_INT,
            comm);

    MPI_Allgather(
            &index,
            1,
            MPI_INT,
            block_index,
            1,
            MPI_INT,
            comm
            );
}

void VectorScatter(
        Vector&  vec_root,
        Vector&  vec_local,
        int     *block_size,
        int     *block_index,
        int      root,
        MPI_Comm comm
){
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Scatterv(
            vec_root.GetData(),
            block_size,
            block_index,
            MPI_DOUBLE,
            vec_local.GetData(),
            block_size[rank],
            MPI_DOUBLE,
            root,
            comm
            );
}

void VectorGather(
        Vector&  vec_root,
        Vector&  vec_local,
        int     *block_size,
        int     *block_index,
        int      root,
        MPI_Comm comm
){
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Gatherv(
            vec_local.GetData(),
            block_size[rank],
            MPI_DOUBLE,
            vec_root.GetData(),
            block_size,
            block_index,
            MPI_DOUBLE,
            root,
            comm);
}


void MatrixScatter(
        Matrix&  mat_root,
        Matrix&  mat_local,
        int     *block_size,
        int     *block_index,
        int      root,
        MPI_Comm comm
){
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Scatterv(
            mat_root.GetData(),
            block_size,
            block_index,
            MPI_DOUBLE,
            mat_local.GetData(),
            block_size[rank],
            MPI_DOUBLE,
            root,
            comm
            );
}
