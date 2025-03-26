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


//Constants
double v=120.0;
double k =2.5;
double w = 4.0;
float alpha = 2*pow(10,-7);
double d = 30.0;

//variables
double L_0=30.0;
double L=30.0;
double p=5.0;
double x=15.0;
double theta= 60 * M_PI / 180;
double phi= 15 * M_PI / 180;
double a=40.0;
double H=100.0;
double n = 5.0;



VecDoub createVars() {
    VecDoub vars(8);
    vars[0] = L_0;   
    vars[1] = L;     
    vars[2] = p;
    vars[3] = x;
    vars[4] = theta;
    vars[5] = phi;
    vars[6] = a;
    vars[7] = H;
    return vars;
}


// functions
VecDoub vecfunc(VecDoub_I &x){
    VecDoub f(8);
    f[0] = x[6]*(cosh(x[3]/x[6])-1.0)-x[2];
    f[1] = 2.0*x[6]*(sinh(x[3]/x[6]))-x[1];
    f[2] = 2.0*x[3]+2.0*k*cos(x[4])-d;
    f[3] = x[2]+k*sin(x[4])-n;
    f[4] = sinh(x[3]/x[6])-tan(x[5]);
    f[5] = (1.0+(v/(w*x[0])))*tan(x[5])-tan(x[4]);
    f[6] = x[0]*(1.0+alpha*x[7])-x[1];
    f[7] = (w*x[0])/(2.0*sin(x[5]))-x[7];
    return f;
    };

void test()
{
    cout<<"Test function to see patterens emerge"<<endl;
    bool check = true;  // Variable to receive convergence status
    n=5.0;
    VecDoub vars = createVars(); // Declare vars outside the loop
    newt(vars, check, vecfunc);  // Pass the function object itself
    // Check if the algorithm converged successfully
    if (check) {    
        cout << "Warning: Possible convergence issues" << endl;
    } else {
        cout << "Solution found: L=" << vars[1] << ", H=" << vars[7] << endl;
    }
    util::print(vars, "vars = ");
    n=2.0;
    newt(vars, check, vecfunc);  // Pass the function object itself
    // Check if the algorithm converged successfully
    if (check) {    
        cout << "Warning: Possible convergence issues" << endl;
    } else {
        cout << "Solution found: L=" << vars[1] << ", H=" << vars[7] << endl;
    }
    util::print(vars, "vars = ");
    n=1.0;
    newt(vars, check, vecfunc);  // Pass the function object itself
    // Check if the algorithm converged successfully
    if (check) {    
        cout << "Warning: Possible convergence issues" << endl;
    } else {
        cout << "Solution found: L=" << vars[1] << ", H=" << vars[7] << endl;
    }
    util::print(vars, "vars = ");

    cout<<"There is a clear pattern in the results"<<endl;
    cout<<"The values of L and L_0 are converging to a certain value: 25"<<endl;
    cout<<"The values of P, theta and phi are roughly halfed in each iteration"<<endl;
    cout<<"The values of a and H are roughly doubled in each iteration"<<endl;
    cout<<"The value of x is converging to 12.5"<<endl;
}


int main() {
    test();
    vector <double> n_values = {5.0,2.0,1.0,0.5,0.2,0.1};
    vector <float> results;
    Bool check = true;  // Variable to receive convergence status
    VecDoub vars; // Declare vars outside the loop
    // Loop over the values of n
    for (int i = 0; i < n_values.size(); i++)
    {
        n = n_values[i];
        L=25.0;
        L_0=25.0;
        p = p/2;
        x=12.5;
        theta= theta/2;
        phi= phi/2;
        a=a*2;
        H=H*2;
        vars = createVars(); // Update vars with initial values
        cout <<"n="<<n<<endl;
        newt(vars, check, vecfunc);  // Pass the function object itself
        // Check if the algorithm converged successfully
        if (check) {    
            cout << "Warning: Possible convergence issues" << endl;
        } else {
            cout << "Solution found: L=" << vars[1] << ", H=" << vars[7] << endl;
        }
        //util::print(vars, "vars = ");
        results.push_back(vars[1]);
        results.push_back(vars[7]);

    }
    // Print the results
    // cout << "Results: " << endl;

    // for (int i = 0; i < results.size(); i++)
    // {
    //     if(i%2==0){
    //         cout << "L: ";
    //     }else{
    //         cout << "H: ";
    //     }
    //     cout << results[i] << endl;
    // }

}