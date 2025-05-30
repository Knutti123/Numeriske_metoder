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
#include <wavelet.h>
#include <weights.h>
#include <zrhqr.h>
#include <utilities.h>
using namespace std;



int main() 
{
    VecDoub xFilip(82); VecDoub yFilip(82);
    ifstream Filip("/home/kristian/Kode/Numeriske_metoder/Numerical-Recipes-master/FilipData.dat");
    for(int i = 0; i < 82; i++) {
        Filip >> yFilip[i];
        Filip >> xFilip[i];
    }
    
    VecDoub xPont(40); VecDoub yPont(40);
    ifstream Pont("/home/kristian/Kode/Numeriske_metoder/Numerical-Recipes-master/PontiusData.dat");
    for(int i = 0; i < 40; i++) {
        Pont >> yPont[i];
        Pont >> xPont[i];
    }
    int N = 40;

    MatDoub A(N,3);
    for(int i = 0; i < N; i++)
    {
        A[i][0] = 1.0;
        A[i][1] = xPont[i];
        A[i][2] = xPont[i]*xPont[i];
    }
    VecDoub b(yPont);
    MatDoub AT = util::Transpose(A);
    MatDoub ATA = AT*A;
    VecDoub ATb = AT*b;

    LUdcmp lu(ATA);
    VecDoub c(3);
    lu.solve(ATb, c);

    Cholesky chol(ATA);
    VecDoub cChol(3);
    chol.solve(ATb, cChol);
    SVD svd(A);
    VecDoub cSVD(3);
    svd.solve(b, cSVD);
    util::print(cSVD, "pontSVD = ");
    util::print(c, "c = ");
    util::print(cChol, "cChol = ");

    MatDoub Afilip(82,11);
    for (int i = 0; i < 82; i++) {
        for (int j = 0; j < 11; j++) {
            Afilip[i][j] = pow(xFilip[i], j);
        }
    }

    
    VecDoub bFilip(yFilip);
    MatDoub ATfilip = util::Transpose(Afilip);
    MatDoub ATAFilip = ATfilip*Afilip;
    VecDoub ATbFilip = ATfilip*bFilip;

    SVD svdFilip(Afilip);
    VecDoub cSVDFilip(11);
    svdFilip.solve(bFilip, cSVDFilip);
    util::print(cSVDFilip, "cSVDFilip = ");

    LUdcmp luFilip(ATAFilip);
    VecDoub cFilip(11);
    luFilip.solve(ATbFilip, cFilip);
    util::print(cFilip, "cFilip = ");

    Cholesky cholFilip(ATAFilip);
    VecDoub cCholFilip(11);
    cholFilip.solve(ATbFilip, cCholFilip);
    
    util::print(cCholFilip, "cCholFilip = ");

    return 0;
}
