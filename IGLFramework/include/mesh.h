#ifndef mesh_H
#define mesh_H

#include <igl/readOFF.h>

#include <ctime>
#include <cstdlib>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace Eigen;
using namespace std;


class mesh {
    public:
    tuple<Matrix3d, Vector3d, double> ICP(MatrixXd Pv, MatrixXd Qv, int step_size);
    SparseMatrix<double> Adjacency_Matrix(MatrixXi F);
    MatrixXd Add_noise(MatrixXd m, double noise_val);
    pair<MatrixXd, MatrixXd> rotate(MatrixXd, double x, double y, double z);
    MatrixXd non_rigid_ICP(MatrixXd Temp_V, MatrixXi Temp_F, MatrixXd Target_V, MatrixXi Target_F);
};
#endif

