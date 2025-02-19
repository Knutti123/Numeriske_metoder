int main() {
    MatDoub A(3, 3);
    A[0][0] = 1.0;
    A[0][1] = 2.0;
    A[0][2] = 3.0;
    A[1][0] = 2.0;
    A[1][1] = -4.0;
    A[1][2] = 6.0;
    A[2][0] = 3.0;
    A[2][1] = -9.0;
    A[2][2] = -3.0;

    VecDoub b(3);
    b[0] = 5.0;
    b[1] = 18.0;
    b[2] = 6.0;

    // Perform LU decomposition
    LUdcmp lu(A);
    VecDoub x(3);
    lu.solve(b, x);

    // Print L and U matrices
    MatDoub L(3, 3), U(3, 3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i > j) {
                L[i][j] = lu.lu[i][j];
                U[i][j] = 0.0;
            } else if (i == j) {
                L[i][j] = 1.0;
                U[i][j] = lu.lu[i][j];
            } else {
                L[i][j] = 0.0;
                U[i][j] = lu.lu[i][j];
            }
        }
    }

    std::cout << "L matrix:\n";
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << L[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "U matrix:\n";
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << U[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Solution x:\n";
    for (int i = 0; i < x.size(); ++i) {
        std::cout << x[i] << std::endl;
    }

    return 0;
}