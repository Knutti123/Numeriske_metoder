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
#include <cassert>

double u(double x, double t)
{
    assert(x>=0.0 && x<=1.0);
    assert(t>=0.0);
    if (t == 0 && x>=0 && x<=1)
        return pow(x,4);
    else if (x==0 && t>0)
        return 0;
    else if (x==1 && t>0)
        return 1;    
}

double f(double x, double t)
{
    return x*(1-x)*cos(t)*exp((-t)/10);
}


double parabolic_pde(int N, int alpha, double f(double x, double y),double u(double x, double y),const double t_end, const double x_target)
{

    auto g = [&](double x) {return u(x,0);};
    double n = N-1;
    double h = 1.0/N;
    double r = alpha*(h/pow(h,2));
    VecDoub A(n);
    VecDoub B(n);
    VecDoub C(n);
    VecDoub R(n);
    VecDoub U(N+1);
    VecDoub Res(n);
    for (int j=0; j<N+1; j++)
    {
        double x = j*h;
        U[j]=u(x,0);
    }
    for (int t=0;t<t_end;t++)
    {
        for (int j=1; j<N; j++)
        {
            double x = j*h;
            A[j-1] = -0.5*r;
            B[j-1] = 1+r;
            C[j-1] = 0.5*r;
            R[j-1] = 0.5*r*U[j-1]+(1+r)*U[j]+0.5*r*U[j+1]+(h/2)*(f(x,t)+f(x,t+1));
        }
        tridag(A,B,C,R,Res);
        for (int j=0; j<N; j++)
        {
            U[j+1] = Res[j];
        }
    }
    util::print(U);
}

int main() {
    int aplha = 1;
    int a=0;
    int b=1;
    int N = 10;
    double t_end = 20;
    double x_target = 0.5;
    double result = parabolic_pde(N, aplha, f, u, t_end, x_target);
    return 0;
}