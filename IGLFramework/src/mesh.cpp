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
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <SymEigsSolver.h>
#include <GenEigsSolver.h>
#include <MatOp/SparseGenMatProd.h>
#include <Eigen/IterativeLinearSolvers>
#include "igl/adjacency_matrix.h"
#include "igl/cat.h"
#include <Eigen/LU>
#include<Eigen/SparseCholesky>
#include <unsupported/Eigen/KroneckerProduct>


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
//    F(0,0) = 0;
//    F(0,1) = 1;
//    F(0,2) = 3;
//    F(1,0) = 1;
//    F(1,1) = 2;
//    F(1,2) = 3;
    Eigen::SparseMatrix<double> A;
    igl::adjacency_matrix(F, A);
    //cout<<A;
    return A;
}

SparseMatrix<double> mesh::Incidence_Matrix(SparseMatrix<double> A) {
    int cols = A.cols();
    assert(cols > 0);

    int rows = A.rows();
    assert(rows > 0);

    assert(rows == cols);
    std::vector<double> Vector_end;
    std::vector<double> Vector_begin;

    int count = 0;
    for (int row = 0; row <= rows; row++){

        for (int col = 0; col <= cols; col++) {
            if(A.coeff(row, col)){
                if(col < row) {
                    Vector_end.push_back(col);
                    Vector_begin.push_back(row);
                    count++;
                }
            }
        }
    }
    cout << "end for" <<endl;

    Eigen::SparseMatrix<double> incidence(cols, Vector_end.size());
    cout << "end init" << Vector_end.size() << " " << count << Vector_begin.size() <<endl;

    for(int i=0;i<Vector_end.size();i++){
        incidence.insert(Vector_end[i],i) = 1;
        incidence.insert(Vector_begin[i],i) = -1;
        //cout << i <<endl;
    }
    return incidence.transpose();
}

SparseMatrix<double> compute_D(MatrixXd V) {
    int nVert = V.rows();
    SparseMatrix<double> D(nVert, nVert * 4);
    for (int i = 0; i < nVert; i++) {
         D.insert(i, i * 4) = V(i, 0);
         D.insert(i, i * 4 + 1) = V(i, 1);
         D.insert(i, i * 4 + 2) = V(i, 2);
         D.insert(i, i * 4 + 3) = 1;
    }
    return D;
}

SparseMatrix<double> concat_rows(MatrixXd A, MatrixXd B) {
    MatrixXd M(A.rows() + B.rows(), A.cols());
    M << A, B;

//    SparseMatrix<double> M(A.rows() + B.rows(), A.cols());
//    M.reserve(A.nonZeros() + B.nonZeros());
//    for(Index c = 0; c < A.rows(); ++c)
//    {
//        for(SparseMatrix<double>::InnerIterator itA(A, c); itA; ++itA)
//            M.insertBack(itA.row(), c) = itA.value();
//        for(SparseMatrix<double>::InnerIterator itB(B, c); itB; ++itB)
//            M.insertBack(itB.row(), c) = itB.value();
//    }
//    M.finalize();
    return M.sparseView();
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
    VectorXd WVec = VectorXd::Ones(nVert);
    MatrixXd W = WVec.asDiagonal();
    MatrixXd init_trans(4, 3);
    MatrixXd X(4 * nVert, 3);
    VectorXd alpha = VectorXd::LinSpaced(20, 100, 10);
    int nAlpha = alpha.rows();

    MatrixXd pre_X;
    MatrixXd U;
    SparseMatrix<double> WD;
    SparseMatrix<double> WU;
    SparseMatrix<double> aMoG;

    R = MatrixXd::Identity(3, 3);
    t = MatrixXd::Zero(1, 3);
    init_trans << R, t;
    X = init_trans.replicate(nVert, 1);

    D = compute_D(Temp_V);

    SparseMatrix<double> A = Adjacency_Matrix(Temp_F);

    SparseMatrix<double> M = Incidence_Matrix(A);
    A.resize(0, 0);

    cout << "M size:" << endl;
    cout << M.rows() << endl;
    cout << M.cols() << endl;

    cout << "G size:" << endl;
    cout << G.rows() << endl;
    cout << G.cols() << endl;

    SparseMatrix<double> MoG(G.rows() * M.rows(), G.cols() * M.cols());
    MatrixXd temp;

    for (int i = 0; i < M.rows(); i++)
    {
        for (int j = 0; j < M.cols(); j++)
        {
            temp = M.coeff(i, j) * G;
            if (temp.norm() != 0) {
                for (int row; row < temp.rows(); row++) {
                    for (int col; col < temp.cols(); col++) {
                        MoG.insert(i * G.rows() + row, j * G.cols() + col) = temp(row, col);
                    }
                }
            }
        }
    }
    //cout << M << endl;
    //MoG = kroneckerProduct(M, G);
    //cout << MoG << endl;

    MatrixXd new_V;

    for (int i = 0; i < nAlpha; i++) {
        double curr_alpha = alpha(i);
        cout << "Alpha:" << endl;
        cout << curr_alpha << endl;
        pre_X = 10 * X;
        //cout << pre_X << endl;
        int iter = 0;
        while ((X - pre_X).norm() >= 0.0001) {
            new_V = D * X;
            U = knnsearch(Target_V, new_V, 1);

            WD = (W * D).sparseView();
            cout << "Built WD" << endl;
            WU = (W * U).sparseView();
            cout << "Built WU" << endl;
            aMoG = curr_alpha * MoG;
            cout << "Built aMoG" << endl;
            SparseMatrix<double> zeros(M.rows() * G.rows(), 3);
            cout << "Matrices Built" << endl;

            SparseMatrix<double> A(aMoG.rows() + WD.rows(), aMoG.cols());

            A = concat_rows(aMoG, WD);

            SparseMatrix<double> B(zeros.rows() + WU.rows(), 3);

            B = concat_rows(zeros, WU);

            pre_X = X;

            SparseMatrix<double> ATA = A.transpose() * A;
            SparseMatrix<double> ATB = A.transpose() * B;
            SimplicialLLT<SparseMatrix<double> > solver;

            solver.compute(ATA);

            X = solver.solve(ATB);

//            cout << "COMPUTE X" << endl;
//
//            cout << "W size:" << endl;
//            cout << W.rows() << endl;
//            cout << W.cols() << endl;
//            cout << "U size:" << endl;
//            cout << U.rows() << endl;
//            cout << U.cols() << endl;
//            cout << "D size:" << endl;
//            cout << D.rows() << endl;
//            cout << D.cols() << endl;
//            cout << "WD size:" << endl;
//            cout << WD.rows() << endl;
//            cout << WD.cols() << endl;
//            cout << "B size:" << endl;
//            cout << B.rows() << endl;
//            cout << B.cols() << endl;
//            cout << "MOG size:" << endl;
//            cout << aMoG.rows() << endl;
//            cout << aMoG.cols() << endl;
        }
//        A.resize(0,0);
//        M.resize(0,0);
//        G.resize(0,0);
//        cout << "X size:" << endl;
//        cout << X.rows() << endl;
//        cout << X.cols() << endl;
    }
    cout << "X rows:" << endl;
    cout << X.rows() << endl;
    cout << "X cols:" << endl;
    cout << X.cols() << endl;
    new_V = D * X;
    cout << "V rows:" << endl;
    cout << new_V.rows() << endl;
    return new_V;

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

