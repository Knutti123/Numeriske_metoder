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
double t = 0.0;
double v1 = 1.0;
double v2 = 2.0;
double v3 = 3.0;



void task_1()
{
    double v1_calc = exp(-t)*cos(v2)+pow(v3,2)-v1;
    double v2_calc = cos(pow(v3, 2)) - v2;
    double v3_calc = cos(t) * exp(-pow(v1, 2)) - v3;

    cout<<cout.precision(6);
    cout<<"v₁(0)"<<v1_calc<<endl;
    cout<<"v₂(0)"<<v2_calc<<endl;
    cout<<"v₃(0)"<<v3_calc<<endl;
}

// Define the differential equations
struct rhs_func {
    void operator() (const Doub x, VecDoub_I &y, VecDoub_O &dydx) {
        dydx[0] = exp(-x)*cos(y[1])+pow(y[2],2)-y[0];  // dv₁/dt
        dydx[1] = cos(pow(y[2], 2)) - y[1];            // dv₂/dt
        dydx[2] = cos(x) * exp(-pow(y[0], 2)) - y[2];  // dv₃/dt
    }
};

void task_2() 
{
    const double t_start = 0.0;
    const double t_end = 5.0;
    const double eps = 1.0e-6;      // Accuracy parameter
        
        // Initial conditions
        VecDoub ystart(3);
        ystart[0] = 1.0;  // v₁
        ystart[1] = 2.0;  // v₂
        ystart[2] = 3.0;  // v₃
        
        // Create the differential equation system
        rhs_func derivs;
        
        // Run for different N values
        vector<int> N_values = {50, 100, 200, 400, 800};
        
        for (int N : N_values) {
            // Step size based on N
            double h = (t_end - t_start) / N;
            
            // Reset initial conditions for each run
            VecDoub y = ystart;
            double t = t_start;
            
            // Temporary vectors for the trapezoidal method
            VecDoub k1(3), k2(3), y_mid(3);
            
            // Apply trapezoidal method with N steps
            for (int i = 0; i < N; i++) {
                // First step: calculate derivatives at current point
                derivs(t, y, k1);
                
                // Predict midpoint values using Euler
                for (int j = 0; j < 3; j++) {
                    y_mid[j] = y[j] + 0.5 * h * k1[j];
                }
                
                // Calculate derivatives at predicted midpoint
                derivs(t + 0.5 * h, y_mid, k2);
                
                // Update solution using trapezoidal rule
                for (int j = 0; j < 3; j++) {
                    y[j] += h * k2[j];
                }
                
                t += h;
            }
            
            // Print results for this N value
            cout << "N = " << N << ":" << endl;
            cout << "  v₁(5) = " << setprecision(8) << y[0] << endl;
            cout << "  v₂(5) = " << setprecision(8) << y[1] << endl;
            cout << "  v₃(5) = " << setprecision(8) << y[2] << endl;
            cout << "  Step size h = " << h << endl << endl;
        }
}




int main() {
    task_1();
    task_2();
    return 0;
}