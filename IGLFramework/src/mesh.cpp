#include "mesh.h"
#include "acq/normalEstimation.h"
#include <nanoflann.hpp>
#include <random>
#include <cmath>
#include <igl/gaussian_curvature.h>
#include <igl/doublearea.h>
#include <exception>
#include <list>
#include <Eigen/Sparse>
#include <SymEigsSolver.h>
#include <GenEigsSolver.h>
#include <MatOp/SparseGenMatProd.h>
#include <Eigen/IterativeLinearSolvers>
#include "igl/adjacency_matrix.h"


using namespace Eigen;
using namespace std;
using namespace nanoflann;
using namespace acq;
using namespace Spectra;


tuple<MatrixXd, MatrixXd, double> nearest_neighbor(MatrixXd Pv, MatrixXd Qv, int sample) {
    const size_t dim = 3;
    const size_t N_q = Qv.rows();
    const size_t N_p = Pv.rows();
    double dist = 0;

    Matrix<double, Dynamic, Dynamic>  mat(N_q, dim);

    for (size_t i = 0; i < N_q; i++)
        for (size_t d = 0; d < dim; d++)
            mat(i,d) = Qv(i,d);

//	cout << mat << endl;
    const size_t num_results = 1;
    typedef KDTreeEigenMatrixAdaptor<Matrix<double, Dynamic, Dynamic> > my_kd_tree_t;

    my_kd_tree_t mat_index(mat, 10 /* max leaf */ );
    mat_index.index->buildIndex();

    // Query point:
    vector<double> query_pt(dim);
    //matching results
    Matrix<double, Dynamic, Dynamic>  P(N_p, dim);
    Matrix<double, Dynamic, Dynamic>  Q(N_p, dim);
    int count = 0;
    for (size_t i = 0; i < Pv.rows(); i+=sample) {
        for (size_t d = 0; d < dim; d++)
            query_pt[d] = Pv(i, d);
        // do a knn search
        vector<size_t> ret_indexes(num_results);
        vector<double> out_dists_sqr(num_results);

        nanoflann::KNNResultSet<double> resultSet(num_results);

        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

//        std::cout << "knnSearch(nn=" << num_results << "): \n";
        if (out_dists_sqr[0] < 0.01) {
            int idx = ret_indexes[0];
            P.row(count) = Pv.row(i);
            Q.row(count) = Qv.row(idx);
            count ++;
            dist += out_dists_sqr[0];
        }
//        for (size_t i = 0; i < num_results; i++)
//            std::cout << "ret_index[" << i << "]=" << ret_indexes[i] << " out_dist_sqr=" << out_dists_sqr[i] << endl;
    }
    P = P.topRows(count);
    dist /= P.rows();
    Q = Q.topRows(count);
    return {P, Q, dist};
}

tuple<Matrix3d, Vector3d, double> mesh::ICP(MatrixXd Pv, MatrixXd Qv, int step_size) {
    Vector3d t;
    MatrixXd P, Q, U, V, R, A;
    MatrixXd P_mean, Q_mean;
    double dist;
    tie(P, Q, dist) = nearest_neighbor(Pv, Qv, step_size);
    //compute the mean
    P_mean = P.colwise().sum()/P.rows();
    Q_mean = Q.colwise().sum()/Q.rows();
    //compute matrix A for SVD
    A = (Q - Q_mean.replicate(Q.rows(), 1)).transpose() * (P - P_mean.replicate(P.rows(), 1));
    //compute SVD
    JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    U = svd.matrixU();
    V = svd.matrixV();
    //compute rotation and translation
    R = V * U.transpose();
    t = P_mean.transpose() - (R * Q_mean.transpose());
    return {R, t, dist};
}


MatrixXd knnsearch(MatrixXd source, MatrixXd target, int sample) {
    const size_t dim = 3;
    const size_t N_q = target.rows();
    const size_t N_p = source.rows();
    double dist = 0;

    Matrix<double, Dynamic, Dynamic>  mat(N_q, dim);

    for (size_t i = 0; i < N_q; i++)
        for (size_t d = 0; d < dim; d++)
            mat(i,d) = target(i,d);

//	cout << mat << endl;
    const size_t num_results = 1;
    typedef KDTreeEigenMatrixAdaptor<Matrix<double, Dynamic, Dynamic> > my_kd_tree_t;

    my_kd_tree_t mat_index(mat, 10 /* max leaf */ );
    mat_index.index->buildIndex();

    // Query point:
    vector<double> query_pt(dim);
    //matching results
    Matrix<double, Dynamic, Dynamic>  results(N_p, dim);
    Matrix<double, Dynamic, Dynamic>  Q(N_p, dim);
    int count = 0;
    for (size_t i = 0; i < source.rows(); i+=sample) {
        for (size_t d = 0; d < dim; d++)
            query_pt[d] = source(i, d);
        // do a knn search
        vector<size_t> ret_indexes(num_results);
        vector<double> out_dists_sqr(num_results);

        nanoflann::KNNResultSet<double> resultSet(num_results);

        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

        int idx = ret_indexes[0];
        results.row(i) = target.row(idx);
    }
    return results;
}

SparseMatrix<double> mesh::Adjacency_Matrix(MatrixXi F) {
//    MatrixXi F(2,3);
//    F(0,0) = 1;
//    F(0,1) = 2;
//    F(0,2) = 4;
//    F(1,0) = 2;
//    F(1,1) = 3;
//    F(1,2) = 4;
    cout<<  F;
    Eigen::SparseMatrix<double> A;
    igl::adjacency_matrix(F, A);
    return A;
}

MatrixXd Incidence_Matrix(MatrixXd A) {

}

SparseMatrix<double> compute_D(MatrixXd V) {
    int nVert = V.rows();
    SparseMatrix<double> sparse_D;
    MatrixXd D(nVert, nVert * 4);
    for (int i = 0; i < nVert; i++) {
         D(i, i * 4) = V(i, 0);
         D(i, i * 4 + 1) = V(i, 1);
         D(i, i * 4 + 2) = V(i, 2);
         D(i, i * 4 + 3) = 1;
    }
    sparse_D = D.sparseView();
    return sparse_D;
}

MatrixXd mesh::non_rigid_ICP(MatrixXd Temp_V, MatrixXi Temp_F, MatrixXd Target_V, MatrixXi Target_F) {

    int nVert = Temp_V.rows();
    int nFace = Temp_F.rows();
    int gamma = 1;
    double dist_err;

    SparseMatrix<double> D;
    MatrixXd R, t;
    MatrixXd G(4, 4);
    G.diagonal() << 1, 1, 1, 1;
    MatrixXd W = MatrixXd::Ones(nVert, 1);
    MatrixXd X(3, 4);
    VectorXd alpha = VectorXd::LinSpaced(20, 100, 10);
    int nAlpha = alpha.rows();

    R = MatrixXd::Identity(3, 3);
    t = MatrixXd::Zero(3, 1);
    X << R, t;

    D = compute_D(Temp_V);

    for (int i = 0; i < nAlpha; i++) {
        double curr_alpha = alpha(i);
        MatrixXd pre_X = 10 * X;
        MatrixXd new_V;

        SparseMatrix<double> A = Adjacency_Matrix(Temp_F);
        //MatrixXd M = Incidence_Matrix(A);
        while ((X - pre_X).norm() >= 0.0001) {
            new_V = D * X;
            MatrixXd U = knnsearch(new_V, Target_V, 1);
        }
    }

//    MatrixXd X = zeros(3, 4 * num_V).transpose();
//    DiagonalMatrix<double, 4> G(1, 1, 1, gamma);
//    MatrixXd M = V;
//    MatrixXd D = compute_D(V);
//    MatrixXd weights = MatrixXd::ones(num_V, 1);
//    tie(R, t, dist_err) = ICP(Target_V, Temp_V, step_size);
//    init_T = R.transpose() << endl << t.transpose();
//    X = init_T.replicate<num_V, 1>();
//    MatrixXd new_V = D * X;
}

