#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace std;
using namespace Eigen;

double errLU(const MatrixXd& A, const VectorXd& b, const VectorXd& sol, Vector2d& xLU)
{
    xLU = A.partialPivLu().solve(b);
    double erroreLU = (sol-xLU).norm()/sol.norm();
    return erroreLU;
}

double errQR(const MatrixXd& A, const VectorXd& b, const VectorXd& sol, Vector2d& xQR)
{
    xQR = A.colPivHouseholderQr().solve(b);
    double erroreQR = (sol-xQR).norm()/sol.norm();
    return erroreQR;
}

int main()
{
    // definizione delle matrici A e dei vettori b per ogni sistema assegnato
    Matrix2d A;
    Vector2d b;
    Vector2d xLU;
    Vector2d xQR;

    Vector2d sol;
    sol << -1.0e+00,-1.0e+00;  // la soluzione è uguale per ogni sistema

    // Sistema 1
    A << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01,
        -9.992887623566787e-01;
    b << -5.169911863249772e-01, 1.672384680188350e-01;

    cout << "Sistema 1:" << endl;

    Vector2d x1LU;
    double errore1lu = errLU(A,b,sol,x1LU);
    cout << "x1LU = " << x1LU << setprecision(6) << " con errore relativo = " << errore1lu << endl;

    Vector2d x1QR;
    double errore1qr = errQR(A,b,sol,x1QR);
    cout << "x1QR = " << x1QR << setprecision(6) << " con errore relativo = " << errore1qr << endl;

    // Sistema 2
    A << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01,
        -8.324762492991313e-01;
    b << -6.394645785530173e-04, 4.259549612877223e-04;

     cout << "Sistema 2:" << endl;

    Vector2d x2LU;
    double errore2lu = errLU(A,b,sol,x2LU);
    cout << "x2LU = " << x2LU << setprecision(6) << " con errore relativo = " << errore2lu << endl;

    Vector2d x2QR;
    double errore2qr = errQR(A,b,sol,x2QR);
    cout << "x2QR = " << x2QR << setprecision(6) << " con errore relativo = " << errore2qr << endl;

    // Sistema 3
    A << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01,
        -8.320502947645361e-01;
    b << -6.400391328043042e-10, 4.266924591433963e-10;

     cout << "Sistema 3:" << endl;

    Vector2d x3LU;
    double errore3lu = errLU(A,b,sol,x3LU);
    cout << "x3LU = " << x3LU << setprecision(6) << " con errore relativo = " << errore3lu << endl;

    Vector2d x3QR;
    double errore3qr = errQR(A,b,sol,x3QR);
    cout << "x3QR = " << x3QR << setprecision(6) << " con errore relativo = " << errore3qr << endl;

    return 0;
}





