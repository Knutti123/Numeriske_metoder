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
using namespace std;
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

// The function to integrate: cos(x^2) * exp(-x)
double eq1(double x) {
    return cos(pow(x, 2)) * exp(-x);
}

// Midpoint integration method
double midpoint(double a, double b, int n, double (*func)(double)) {
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double x_mid = a + (i + 0.5) * h;
        sum += func(x_mid);
    }
    return h * sum;
}

// Calculate Richardson extrapolation using Midpnt
void richardsonExtrapolation(double a, double b, int maxLevel, int order) {
    std::vector<std::vector<double>> A(maxLevel + 1, std::vector<double>(maxLevel + 1, 0.0));
    std::vector<double> diffs(maxLevel + 1, 0.0);

    // Define the function to integrate
    auto func = [](double x) { return cos(pow(x, 2)) * exp(-x); };

    // Calculate initial approximations with increasing number of intervals using Midpnt
    for (int i = 1; i <= maxLevel; i++) {
        int n = pow(2, i - 1);
        Midpnt<decltype(func)> midpnt(func, a, b);
        for (int j = 0; j < n; j++) {
            midpnt.next();
        }
        A[i][1] = midpnt.s;
    }

    // Print header
    std::cout << "MidPoint result:  h1/h2 = 2" << std::endl;
    std::cout << std::setw(5) << "i"
              << std::setw(15) << "A(h_i)"
              << std::setw(20) << "A(h_(i-1)) - A(h_i)"
              << std::setw(15) << "alp^k"
              << std::setw(15) << "Rich-error"
              << std::setw(15) << "f-calculations"
              << std::setw(15) << "order-estimate"
              << std::endl;

    // Calculate and print first row (no differences available yet)
    std::cout << std::setw(5) << 1
              << std::setw(15) << std::fixed << std::setprecision(6) << A[1][1]
              << std::endl;

    // Calculate difference for second row
    diffs[2] = A[1][1] - A[2][1];

    // Print second row (no ratio available yet)
    std::cout << std::setw(5) << 2
              << std::setw(15) << std::fixed << std::setprecision(6) << A[2][1]
              << std::setw(20) << std::scientific << std::setprecision(8) << diffs[2]
              << std::endl;

    // Process remaining rows
    for (int i = 3; i <= maxLevel; i++) {
        // Calculate differences for current row
        diffs[i] = A[i - 1][1] - A[i][1];

        // Calculate ratio of consecutive differences (alp^k)
        double alpha_k = 0.0;
        if (fabs(diffs[i - 1]) > 1e-20) {  // Avoid division by very small numbers
            alpha_k = diffs[i - 1] / diffs[i];
        }

        // Calculate Richardson error estimate
        double rich_error = 0.0;
        if (i > 2) {
            rich_error = (A[i][1] - A[i - 1][1]) / (pow(2, order) - 1);
        }

        // Calculate order estimate
        double order_estimate = 0.0;
        if (i > 3 && fabs(diffs[i - 1]) > 1e-20) {
            order_estimate = log(fabs(diffs[i - 1] / diffs[i])) / log(2.0);
        }

        // Number of function calculations
        int f_calcs = pow(2, i - 1);

        // Print the row
        std::cout << std::setw(5) << i
                  << std::setw(15) << std::fixed << std::setprecision(6) << A[i][1]
                  << std::setw(20) << std::scientific << std::setprecision(6) << diffs[i]
                  << std::setw(15) << std::fixed << std::setprecision(6) << alpha_k
                  << std::setw(15) << std::scientific << std::setprecision(6) << rich_error
                  << std::setw(15) << f_calcs
                  << std::setw(15) << std::fixed << std::setprecision(6) << order_estimate
                  << std::endl;
    }

    // Print the final result
    std::cout << "Final result: " << std::fixed << std::setprecision(6) << A[maxLevel][maxLevel] << std::endl;
}

int main() {
    double a = 0.0;    // Lower bound of integration
    double b = 1.0;    // Upper bound of integration
    int order = 2;   //
    int maxLevel = 16; // Maximum level for Richardson extrapolation
    
    std::cout << "cos(pow(x,2))*exp(-x)" << std::endl;
    richardsonExtrapolation(a, b, maxLevel,order);
    
    return 0;
}
