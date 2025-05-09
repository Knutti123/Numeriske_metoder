#include <iostream>
#include <nr3.h>
#include <adapt.h>
#include <amebsa.h>
#include <amoeba.h>
#include <anneal.h>
#include <arithcode.h>
#include <asolve.h>
#include <banded.h>
#include <besselfrac.h>
#include <bessel.h>
#include <calendar.h>
#include <chebyshev.h>
#include <cholesky.h>
#include <circumcircle.h>
#include <cisi.h>
#include <convlv.h>
#include <correl.h>
#include <dawson.h>
#include <decchk.h>
#include <delaunay.h>
#include <derule.h>
#include <deviates.h>
#include <dfridr.h>
#include <dftintegrate.h>
#include <difeq.h>
#include <distributions.h>
#include <dynpro.h>
#include <eclass.h>
#include <eigen_sym.h>
#include <eigen_unsym.h>
#include <elliptint.h>
#include <erf.h>
#include <expint.h>
#include <fasper.h>
#include <fermi.h>
#include <fitab.h>
#include <fit_examples.h>
#include <fitexy.h>
#include <fitlin.h>
#include <fitmed.h>
#include <fitmrq.h>
#include <fitsvd.h>
#include <fourfs.h>
#include <fourier.h>
#include <fourier_ndim.h>
#include <fred2.h>
#include <fred_singular.h>
#include <frenel.h>
#include <gamma.h>
#include <gaumixmod.h>
#include <gaussj.h>
#include <gauss_wgts2.h>
#include <gauss_wgts.h>
#include <hashall.h>
#include <hash.h>
#include <hmm.h>
#include <huffcode.h>
#include <hypgeo.h>
#include <icrc.h>
#include <igray.h>
#include <incgammabeta.h>
#include <interior.h>
#include <interp_1d.h>
#include <interp_2d.h>
#include <interp_curve.h>
#include <interp_laplace.h>
#include <interp_linear.h>
#include <interp_rbf.h>
#include <iqagent.h>
#include <kdtree.h>
#include <kmeans.h>
#include <krig.h>
#include <ksdist.h>
#include <kstests_2d.h>
#include <kstests.h>
#include <linbcg.h>
#include <linpredict.h>
#include <ludcmp.h>
#include <machar.h>
#include <markovgen.h>
#include <mcintegrate.h>
#include <mcmc.h>
#include <mgfas.h>
#include <mglin.h>
#include <mins.h>
#include <mins_ndim.h>
#include <miser.h>
#include <mnewt.h>
#include <moment.h>
#include <mparith.h>
#include <multinormaldev.h>
#include <odeint.h>
#include <pade.h>
#include <pcshft.h>
#include <period.h>
#include <phylo.h>
#include <plegendre.h>
#include <pointbox.h>
#include <polcoef.h>
#include <polygon.h>
#include <poly.h>
#include <primpolytest.h>
#include <psplotexample.h>
#include <psplot.h>
#include <qgaus.h>
#include <qotree.h>
#include <qrdcmp.h>
#include <qroot.h>
#include <quad3d.h>
#include <quadrature.h>
#include <quadvl.h>
#include <quasinewton.h>
#include <ran.h>
#include <ranpt.h>
#include <ratlsq.h>
#include <rebin.h>
#include <rk4.h>
#include <romberg.h>
#include <roots.h>
#include <roots_multidim.h>
#include <roots_poly.h>
#include <savgol.h>
#include <scrsho.h>
#include <selip.h>
#include <series.h>
#include <shoot.h>
#include <shootf.h>
//#include <simplex.h>
#include <sobseq.h>
#include <solvde.h>
#include <sor.h>
#include <sort.h>
#include <sparse.h>
#include <spectrum.h>
#include <sphcirc.h>
#include <sphfpt.h>
#include <sphoot.h>
#include <stattests.h>
#include <stepperbs.h>
#include <stepperdopr5.h>
#include <stepperdopr853.h>
#include <stepper.h>
#include <stepperross.h>
#include <steppersie.h>
#include <stepperstoerm.h>
#include <stiel.h>
#include <stochsim.h>
#include <stringalign.h>
#include <svd.h>
#include <svm.h>
#include <toeplz.h>
#include <tridag.h>
#include <vander.h>
#include <vegas.h>
#include <voltra.h>
#include <voronoi.h>
#include <utilities.h>
#include <wavelet.h>
#include <weights.h>
#include <zrhqr.h>
#include <tridag.h>
using namespace std;
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

vector<double> initializeGrid(double x0, double x2, int N) {
    vector<double> x(N + 2);
    double dx = (x2 - x0) / (N + 1);
    for (int i = 0; i <= N + 1; i++) {
        x[i] = x0 + i * dx;
    }
    return x;
}

// Initialize solution with linear interpolation between boundary conditions
vector<double> initializeSolution(const vector<double>& x, double y_start, double y_end) {
    int N = x.size() - 2;
    vector<double> y(N + 2);
    y[0] = y_start;
    y[N + 1] = y_end;
    for (int i = 1; i <= N; i++) {
        y[i] = y[0] + (y[N + 1] - y[0]) * (x[i] - x[0]) / (x[N + 1] - x[0]);
    }
    return y;
}

// Set up and solve the tridiagonal system for one Newton iteration
double performNewtonIteration(vector<double>& y, const vector<double>& x, double dx) {
    int N = x.size() - 2;
    double dx2 = dx * dx;
    VecDoub a(N), b(N), c(N), r(N);
    
    // Fill the Jacobian and residual vector
    for (int i = 1; i <= N; i++) {
        // Define derivatives for neighboring points
        double yp_forward = (i < N) ? (y[i+1] - y[i]) / dx : (y[N+1] - y[N]) / dx;
        double yp_backward = (y[i] - y[i-1]) / dx;
        double yp_central = (yp_forward + yp_backward) / 2;
        
        // Define residual phi
        double residual = (y[i+1] - 2.0*y[i] + y[i-1]) / dx2 - 2.0*x[i] - sin(yp_central) + cos(y[i]);
        r[i-1] = -residual;
        
        // Jacobian components
        if (i > 1) a[i-1] = 1.0/dx2 - 0.5 * cos(yp_central) / dx;
        b[i-1] = -2.0/dx2 + sin(y[i]);
        if (i < N) c[i-1] = 1.0/dx2 + 0.5 * cos(yp_central) / dx;
    }
    
    // Solve the system J * Δy = -phi
    VecDoub delta_y(N);
    tridag(a, b, c, r, delta_y);
    
    // Update solution and return max residual
    double maxRes = 0.0;
    for (int i = 1; i <= N; i++) {
        y[i] += delta_y[i-1];
        maxRes = max(maxRes, fabs(delta_y[i-1]));
    }
    
    return maxRes;
}

// Print the solution
void printResults(const vector<double>& x, const vector<double>& y) {
    cout << "\nResults:\n";
    cout << "x\ty\n";
    cout << "----------------\n";
    for (size_t i = 0; i < x.size(); i++) {
        cout << fixed << setprecision(4) << x[i] << "\t" << y[i] << endl;
    }
}

int main() {
    // Parameters
    const int N = 100;  // Grid points (excluding boundaries)
    const double x0 = 0.0;
    const double x2 = 2.0;
    const double y_start = 0.0;  // y(0) = 0
    const double y_end = 1.0;    // y(2) = 1
    
    // Set up grid and initial solution
    vector<double> x = initializeGrid(x0, x2, N);
    vector<double> y = initializeSolution(x, y_start, y_end);
    
    double dx = (x2 - x0) / (N + 1);  // Step size
    
    // Newton iteration
    const int maxIter = 100;
    const double tol = 1e-8;
    int iter = 0;
    double maxRes = 1.0;
    
    while (maxRes > tol && iter < maxIter) {
        maxRes = performNewtonIteration(y, x, dx);
        
        iter++;
        cout << "Iteration " << iter << ": max residual = " << maxRes << endl;
    }
    
    // Output results
    printResults(x, y);

    cout << "\nValue at y(1): " << y[N/2 + 1] << endl;
    cout << "Converged in " << iter << " iterations." << endl;
    
    return 0;
}