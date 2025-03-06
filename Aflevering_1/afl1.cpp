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
#include <math.h>
using namespace std;



int main() 
{
    //Setup of data
    //Matrix A setup
        ifstream ex1("/home/kristian/Kode/Numeriske_metoder/Numerical-Recipes-master/Ex1A.dat");
        double L, W;
        ex1 >> L;
        ex1 >> W;
        //cout << L << " " << W << endl;
        MatDoub ex1A(L,W);
        for(int i = 0; i < L; i++) {
                for(int j = 0; j < W; j++) {
                        ex1 >> ex1A[i][j];
                }
        }
        //Matrix B setup (vector)
        ifstream ex1b("/home/kristian/Kode/Numeriske_metoder/Numerical-Recipes-master/Ex1b.dat");
        double L_B, W_B;
        ex1b >> L_B;
        ex1b >> W_B;
        VecDoub b(L);
        //cout << L_B << " " << W_B << endl;
        for(int i = 0; i < L_B; i++) {
                ex1b >> b[i];
        }
        //Task 1
        SVD svd(ex1A);
        VecDoub x(W);
        svd.solve(b,x);
        //Print the diagonal elements in matrix W
        util::print(svd.w);
        cout<<""<<endl;
        //Task 2
        //Print the solution x
        util::print(x);
        //Task 3
        //State an estimate of the accuracy of the solution x
        //Residual error
        double Residual_err = 0;
        double normB = 0;
        //This calculates the residual error
        for (int i = 0; i < L; i++) {
            double Ax_i = 0;
            for (int j = 0; j < W; j++) {
                Ax_i += ex1A[i][j] * x[j];
            }
        //we store the data from the row in the residual error
        Residual_err += pow(b[i] - Ax_i, 2);
        normB += pow(b[i], 2);
        }
        //We take the square root of the residual error and divide it by the square root of the norm of B
        Residual_err = sqrt(Residual_err)/sqrt(normB);
        cout << "Residual error = " << Residual_err << endl;
        //random fitting
        double random_fit = sqrt((L - W) / L);
        cout << "Random fitting = " << random_fit << endl;
        //The soultion x has a good accuracy according to the residual error in regards to the random fitting

        //Task 4
        //Compute and state the residual vector r = b - Ax
        VecDoub r(L);
        for (int i = 0; i < L; i++) {
            double Ax_i = 0;
            for (int j = 0; j < W; j++) {
                Ax_i += ex1A[i][j] * x[j];
            }
            r[i] = Ax_i - b[i];
        }
        //Print the residual vector r
        cout<<"Residual vector r = b - Ax"<<endl;
        util::print(r);  
        //Task 5
        //Compute the new sigma_i and then design the new matrix A and new B vector. State A[0][0] and B[6].
        VecDoub Sigma_I(L);
        Doub Delta =1.0;
        for (int i = 0; i < L; i++) {
            Sigma_I[i] = max(Delta, abs(r[i]));
        }
        //Design the new matrix A and new B vector
        MatDoub new_A(L,W);
        VecDoub new_B(L);
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < W; j++) {
                new_A[i][j] = ex1A[i][j] / Sigma_I[i];
            }
            new_B[i] = b[i] / Sigma_I[i];
        }
        //Print A[0][0] and B[6]
        cout << "A[0][0] = " << new_A[0][0] << endl;
        cout << "B[6] = " << new_B[6] << endl;
        //Task 6
        //Compute and use the Singular Value Decomposition to compute the solution x to Ax=b with the new matrix A and new vector B
        SVD svd_new(new_A);
        VecDoub x_new(W);
        svd_new.solve(new_B,x_new);
        //Print the solution x
        cout<<"Solution x to Ax=b with the new matrix A and new vector B"<<endl;
        util::print(x_new);

}
