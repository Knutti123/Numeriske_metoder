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


        template<class T>
        Doub rtbis_test(T &func, const Doub x1, const Doub x2, const Doub xacc,vector<Doub> &x_k, vector<Doub> &d_k) {
        const Int JMAX = 50;
        Doub dx, xmid, rtb;
        Doub f = func(x1);
        Doub fmid = func(x2);
        if (f * fmid >= 0.0) throw("Root must be bracketed for bisection in rtbis");
        rtb = f < 0.0 ? (dx = x2 - x1, x1) : (dx = x1 - x2, x2);
        double x_prev = rtb;
        for (Int j = 0; j < JMAX; j++) {
        fmid = func(xmid = rtb + (dx *= 0.5));
        //x_k.push_back(rtb);
        if (fmid <= 0.0) rtb = xmid;
        x_k.push_back(rtb);
        d_k.push_back(abs(x_prev - rtb));
        if (abs(dx) < xacc || fmid == 0.0) return rtb;
        x_prev = rtb; 
        }
        throw("Too many bisections in rtbis");
        }

        template<class T>
        Doub rtflsp(T &func, const Doub x1, const Doub x2, const Doub xacc,vector<Doub> &x_k, vector<Doub> &d_k) {
            const Int MAXIT = 30;
            Doub xl, xh, del;
            Doub fl = func(x1);
            Doub fh = func(x2);
            if (fl * fh > 0.0) throw("Root must be bracketed in rtflsp");
            if (fl < 0.0) {
                xl = x1;
                xh = x2;
            } else {
                xl = x2;
                xh = x1;
                SWAP(fl, fh);
            }
            Doub dx = xh - xl;
            Doub x_prev = xl; //To calculate d_k
            for (Int j = 0; j < MAXIT; j++) {
                Doub rtf = xl + dx * fl / (fl - fh);
                Doub f = func(rtf);
                if (f < 0.0) {
                    del = xl - rtf;
                    xl = rtf;
                    fl = f;
                } else {
                    del = xh - rtf;
                    xh = rtf;
                    fh = f;
                }
                dx = xh - xl;
                //Store information
                x_k.push_back(rtf);
                d_k.push_back(abs(x_prev - rtf));
                x_prev = rtf;
                if (abs(del) < xacc || f == 0.0) return rtf;
            }
            throw("Maximum number of iterations exceeded in rtflsp");
        }

        int main() 
        {
                // Create a function
                struct Func {
                        double operator()(double x) {
                                return x - cos(x);
                        }
                };
                Func func;

                // Print the table header
                cout << "k" << "\t" << "x_k" << "\t" << "d_k" << endl;

                // Variables for iteration
                const double xacc = pow(10, -16);
                vector <double> x_k_vec;
                vector <double> d_k_vec;
                double root = rtbis_test(func, 0, 1, xacc, x_k_vec, d_k_vec);
                for (size_t i = 0; i < x_k_vec.size(); i++)
                {
                        cout<<i<<"\t"<<x_k_vec[i]<<"\t"<<d_k_vec[i]<<endl;
                }
                cout << "k" << "\t" << "x_k" << "\t" << "d_k" << endl;
                vector <double> x_k_vec2;
                vector <double> d_k_vec2;
                root = rtflsp(func, 0, 1, xacc, x_k_vec2, d_k_vec2);
                for (size_t i = 0; i < x_k_vec2.size(); i++)
                {
                        cout<<i<<"\t"<<x_k_vec2[i]<<"\t"<<d_k_vec2[i]<<endl;
                }
                return 0;
                

        }