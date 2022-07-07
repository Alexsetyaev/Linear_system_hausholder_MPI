#include <math.h>
#include <mpi.h>
#include <ctime>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

double norma(const std::vector<double>& vect, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += vect[i] * vect[i];
  }
  return sqrt(sum);
}

void print_matrix(const std::vector<std::vector<double> >& matr) {
  for (int i = 0; i < matr.size(); ++i) {
    for (int j = 0; j < matr[0].size(); ++j) {
      std::cout << matr[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void hausholder(std::vector<double>& x,
                const std::vector<std::vector<double> >& matr,
                int i,
                int n,
                int rank,
                int size) {
  int index = i / size;
  for (int j = i; j < n; ++j) {
    x[j - i] = matr[j][index];
  }
  double x0 = x[0];
  double norm = norma(x, n - i);
  x[0] -= norm;
  norm = sqrt(norm * norm - x0 * x0 + (x0 - norm) * (x0 - norm));
  for (int j = 0; j < n - i; ++j) {
    x[j] /= norm;
  }
}

void matrix_mult(const std::vector<double>& x,
                 std::vector<std::vector<double> >& matrix,
                 int k,
                 int n,
                 int rank,
                 int size) {
  int index = 0;
  for (int i = rank; i < n; i += size) {
    index = i / size;
    if (i >= k) {
      double sum = 0.0;
      for (int j = 0; j < n - k; j++) {
        sum += x[j] * matrix[k + j][index];
      }
      sum *= 2.0;
      for (int j = 0; j < n - k; j++) {
        matrix[k + j][index] -= sum * x[j];
      }
    }
  }
}

void vector_mult(const std::vector<double>& x,
                 std::vector<double>& matrix,
                 int k) {
  int n = matrix.size();
  double sum = 0.0;
  for (int j = 0; j < n - k; ++j) {
    sum += 2 * x[j] * matrix[j + k];
  }
  for (int j = 0; j < n - k; ++j) {
    matrix[k + j] -= sum * x[j];
  }
}

double error_norm(std::vector<double> x) {
  for (int i = 0; i < x.size(); ++i) {
    x[i] -= 1.0;
  }
  return norma(x, x.size());
}

int main(int argc, char** argv) {
  int rank, size;
  MPI_Status status;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int n = 6000;
  std::vector<std::vector<double> > A;
  std::vector<double> b;
  std::vector<double> h(n, 0.0);
  std::vector<double> X(n, 0.0);

  double T1_start, T1_end, T2_start, T2_end;

  for (int i = 0; i < n; ++i) {
    std::vector<double> tmp;
    double sum = 0.0;
    for (int j = rank; j < n; j += size) {
      tmp.push_back(double((i + 1) / (j + 1) + 10));
    }
    A.push_back(tmp);
  }
  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += double((i + 1) / (j + 1) + 10);
    }
    b.push_back(sum);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  T1_start = MPI_Wtime();
  for (int i = 0; i < n - 1; ++i) {
    if (rank == i % size) {
      hausholder(h, A, i, n, rank, size);
    }
    MPI_Bcast(&h[0], h.size(), MPI_DOUBLE, i % size, MPI_COMM_WORLD);

    matrix_mult(h, A, i, n, rank, size);
    vector_mult(h, b, i);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  T1_end = MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);
  T2_start = MPI_Wtime();

  for (int i = n - 1; i >= 0; --i) {
    double sum = 0.0;
    if (rank == 0) {
      sum += b[i];
    }
    for (int j = rank; j < n; j += size) {
      if (j >= i + 1) {
        sum -= A[i][j / size] * X[j];
      }
    }
    double global_sum = 0.0;
    MPI_Reduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, i % size,
               MPI_COMM_WORLD);
    if (i % size == rank) {
      X[i] = global_sum / A[i][i / size];
    }
    MPI_Bcast(&X[i], 1, MPI_DOUBLE, i % size, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  T2_end = MPI_Wtime();

  std::cout << "T1: " << T1_end - T1_start << ", T2: " << T2_end - T2_start
            << ", Tsum: " << T1_end - T1_start + T2_end - T2_start
            << ", Error norm: " << error_norm(X) << std::endl;

  MPI_Finalize();

  return 0;
}
