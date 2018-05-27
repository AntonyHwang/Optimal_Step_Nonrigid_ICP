#include "acq/normalEstimation.h"
#include "acq/decoratedCloud.h"
#include "acq/cloudManager.h"

#include "nanogui/formhelper.h"
#include "nanogui/screen.h"

#include "igl/readOFF.h"
#include "igl/viewer/Viewer.h"
#include "mesh.h"

#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;
namespace acq {

/** \brief                      Re-estimate normals of cloud \p V fitting planes
 *                              to the \p kNeighbours nearest neighbours of each point.
 * \param[in ] kNeighbours      How many neighbours to use (Typiclaly: 5..15)
 * \param[in ] vertices         Input pointcloud. Nx3, where N is the number of points.
 * \param[in ] maxNeighbourDist Maximum distance between vertex and neighbour.
 * \param[out] viewer           The viewer to show the normals at.
 * \return                      The estimated normals, Nx3.
 */
    NormalsT
    recalcNormals(
            int                 const  kNeighbours,
            CloudT              const& vertices,
            float               const  maxNeighbourDist
    ) {
        NeighboursT const neighbours =
                calculateCloudNeighbours(
                        /* [in]        cloud: */ vertices,
                        /* [in] k-neighbours: */ kNeighbours,
                        /* [in]      maxDist: */ maxNeighbourDist
                );

        // Estimate normals for points in cloud vertices
        NormalsT normals =
                calculateCloudNormals(
                        /* [in]               Cloud: */ vertices,
                        /* [in] Lists of neighbours: */ neighbours
                );

        return normals;
    } //...recalcNormals()

    void setViewerNormals(
            igl::viewer::Viewer      & viewer,
            CloudT              const& vertices,
            NormalsT            const& normals
    ) {
        // [Optional] Set viewer face normals for shading
        //viewer.data.set_normals(normals);

        // Clear visualized lines (see Viewer.clear())
        viewer.data.lines = Eigen::MatrixXd(0, 9);

        // Add normals to viewer
        viewer.data.add_edges(
                /* [in] Edge starting points: */ vertices,
                /* [in]       Edge endpoints: */ vertices + normals * 0.01, // scale normals to 1% length
                /* [in]               Colors: */ Eigen::Vector3d::Zero()
        );
    }

} //...ns acq

MatrixXd read_vertex_file(char file_path[]) {
    double x, y, z;
    MatrixXd V(0, 3);
    ifstream infile(file_path);
    string line;
    while (getline(infile, line))
    {
        Vector3d temp;
        infile >> x >> y >> z;
        temp << x, y, z;
        V.conservativeResize(V.rows() + 1, V.cols());
        V.row(V.rows() - 1) = temp.transpose();
    }
    infile.close();
    cout << "first V: " << V.row(0) << endl;
    cout << "last V: " << V.row(V.rows() - 1) << endl;
    return V;
}

MatrixXi read_face_file(char file_path[]) {
    int v1, v2, v3;
    MatrixXi F(0, 3);
    ifstream infile(file_path);
    string line;
    while (getline(infile, line))
    {
        Vector3i temp;
        infile >> v1 >> v2 >> v3;
        temp << v1 - 1, v2 - 1, v3 - 1;
        F.conservativeResize(F.rows() + 1, F.cols());
        F.row(F.rows() - 1) = temp.transpose();
    }
    infile.close();
    cout << "first F: " << F.row(0) << endl;
    cout << "last F: " << F.row(F.rows() - 1) << endl;
    return F;
}

int main(int argc, char *argv[]) {

    // Pointcloud vertices, N rows x 3 columns.
    Eigen::MatrixXd V;
    // Face indices, M x 3 integers referring to V.
    Eigen::MatrixXi F;
    // How many neighbours to use for normal estimation, shown on GUI.
    int kNeighbours = 10;
    // Maximum distance between vertices to be considered neighbours (FLANN mode)
    float maxNeighbourDist = 0.15; //TODO: set to average vertex distance upon read

    // Dummy enum to demo GUI
    enum Orientation { Up=0, Down, Left, Right } dir = Up;
    // Dummy variable to demo GUI
    bool boolVariable = true;
    // Dummy variable to demo GUI
    float floatVariable = 0.1f;

//    // Load a mesh in OFF format
//    std::string mesh1 = "../data/face.off";
//    std::string mesh2 = "../data/socket2.off";
//    if (argc > 1) {
//        mesh1 = std::string(argv[1]);
//        if (mesh1.find(".off") == std::string::npos) {
//            std::cerr << "Only ready for  OFF files for now...\n";
//            return EXIT_FAILURE;
//        }
//    } else {
//        std::cout << "Usage: iglFrameWork <path-to-off-mesh.off>." << "\n";
//    }
//    if (argc > 1) {
//        mesh2 = std::string(argv[1]);
//        if (mesh2.find(".off") == std::string::npos) {
//            std::cerr << "Only ready for  OFF files for now...\n";
//            return EXIT_FAILURE;
//        }
//    } else {
//        std::cout << "Usage: iglFrameWork <path-to-off-mesh.off>." << "\n";
//    }

    // Visualize the mesh in a viewer
    igl::viewer::Viewer viewer;
    {
        // Don't show face edges
        viewer.core.show_lines = false;
    }

    // Store cloud so we can store normals later
    acq::CloudManager cloudManager;
    // Read mesh from meshPath
    {
        // Pointcloud vertices, N rows x 3 columns.
        Eigen::MatrixXd Temp_V, Target_V;
        // Face indices, M x 3 integers referring to V.
        Eigen::MatrixXi Temp_F, Target_F;

        char s_v[] = "../data/source_vertex.txt";
        char s_f[] = "../data/source_face.txt";
        Temp_V = read_vertex_file(s_v);
        Temp_F = read_face_file(s_f);

//        // Read mesh1
//        igl::readOFF(mesh1, Temp_V, Temp_F);
//        // Check, if any vertices read
//        if (Temp_V.rows() <= 0) {
//            std::cerr << "Could not read mesh at " << mesh1
//                      << "...exiting...\n";
//            return EXIT_FAILURE;
//        } //...if vertices read

        // Store read vertices and faces
        cloudManager.addCloud(acq::DecoratedCloud(Temp_V, Temp_F));

        char t_v[] = "../data/target_vertex.txt";
        char t_f[] = "../data/target_face.txt";
        Target_V = read_vertex_file(t_v);
        Target_F = read_face_file(t_f);

//        // Read mesh1
//        igl::readOFF(mesh2, Target_V, Target_F);
//        // Check, if any vertices read
//        if (Target_V.rows() <= 0) {
//            std::cerr << "Could not read mesh at " << mesh2
//                      << "...exiting...\n";
//            return EXIT_FAILURE;
//        } //...if vertices read

        // Store read vertices and faces
        cloudManager.addCloud(acq::DecoratedCloud(Target_V, Target_F));


        MatrixXd V(Temp_V.rows() + Target_V.rows(), Temp_V.cols());
        V << Temp_V, Target_V;
        MatrixXi F(Temp_F.rows() + Target_F.rows(),Temp_F.cols());
        F << Temp_F, (Target_F.array() + Temp_V.rows());

        // Store(overwrite) new vertices and faces:
        viewer.data.clear();
        cloudManager.setCloud(acq::DecoratedCloud(V, F),0);
        cloudManager.setCloud(acq::DecoratedCloud(Temp_V,Temp_F),1);
        cloudManager.setCloud(acq::DecoratedCloud(Target_V,Target_F),2);
        viewer.data.clear();
        // Show mesh
        viewer.data.set_mesh(
                cloudManager.getCloud(0).getVertices(),
                cloudManager.getCloud(0).getFaces()
        );

        // Set color for each Mesh.
        Eigen::MatrixXd Color(F.rows(),3);
        Color<< Eigen::RowVector3d(0.99,0.2,0.6).replicate(Temp_F.rows(),1), //pink
                Eigen::RowVector3d(1.0,0.7,0.2).replicate(Target_F.rows(),1); //yellow 0.99,0.2,0.6

        viewer.data.set_colors(Color);

    } //...read mesh

    // Extend viewer menu using a lambda function
    viewer.callback_init =
            [
                    &cloudManager, &kNeighbours, &maxNeighbourDist,
                    &floatVariable, &boolVariable, &dir, &V, &F
            ] (igl::viewer::Viewer& viewer)
            {
                // Add an additional menu window
                viewer.ngui->addWindow(Eigen::Vector2i(900,10), "Acquisition3D");

                // Add new group
                viewer.ngui->addGroup("Nearest neighbours (pointcloud, FLANN)");

                // Add k-neighbours variable to GUI
                viewer.ngui->addVariable<int>(
                        /* Displayed name: */ "k-neighbours",

                        /*  Setter lambda: */ [&] (int val) {
                            // Store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // Store new value
                            kNeighbours = val;

                            // Recalculate normals for cloud and update viewer
                            cloud.setNormals(
                                    acq::recalcNormals(
                                            /* [in]      K-neighbours for FLANN: */ kNeighbours,
                                            /* [in]             Vertices matrix: */ cloud.getVertices(),
                                            /* [in]      max neighbour distance: */ maxNeighbourDist
                                    )
                            );

                            // Update viewer
                            acq::setViewerNormals(
                                    /* [in, out] Viewer to update: */ viewer,
                                    /* [in]            Pointcloud: */ cloud.getVertices(),
                                    /* [in] Normals of Pointcloud: */ cloud.getNormals()
                            );
                        }, //...setter lambda

                        /*  Getter lambda: */ [&]() {
                            return kNeighbours; // get
                        } //...getter lambda
                ); //...addVariable(kNeighbours)

                // Add maxNeighbourDistance variable to GUI
                viewer.ngui->addVariable<float>(
                        /* Displayed name: */ "maxNeighDist",

                        /*  Setter lambda: */ [&] (float val) {
                            // Store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // Store new value
                            maxNeighbourDist = val;

                            // Recalculate normals for cloud and update viewer
                            cloud.setNormals(
                                    acq::recalcNormals(
                                            /* [in]      K-neighbours for FLANN: */ kNeighbours,
                                            /* [in]             Vertices matrix: */ cloud.getVertices(),
                                            /* [in]      max neighbour distance: */ maxNeighbourDist
                                    )
                            );

                            // Update viewer
                            acq::setViewerNormals(
                                    /* [in, out] Viewer to update: */ viewer,
                                    /* [in]            Pointcloud: */ cloud.getVertices(),
                                    /* [in] Normals of Pointcloud: */ cloud.getNormals()
                            );
                        }, //...setter lambda

                        /*  Getter lambda: */ [&]() {
                            return maxNeighbourDist; // get
                        } //...getter lambda
                ); //...addVariable(kNeighbours)

                // Add a button for estimating normals using FLANN as neighbourhood
                // same, as changing kNeighbours
                viewer.ngui->addButton(
                        /* displayed label: */ "Estimate normals (FLANN)",

                        /* lambda to call: */ [&]() {
                            // store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // calculate normals for cloud and update viewer
                            cloud.setNormals(
                                    acq::recalcNormals(
                                            /* [in]      k-neighbours for flann: */ kNeighbours,
                                            /* [in]             vertices matrix: */ cloud.getVertices(),
                                            /* [in]      max neighbour distance: */ maxNeighbourDist
                                    )
                            );

                            // update viewer
                            acq::setViewerNormals(
                                    /* [in, out] viewer to update: */ viewer,
                                    /* [in]            pointcloud: */ cloud.getVertices(),
                                    /* [in] normals of pointcloud: */ cloud.getNormals()
                            );
                        } //...button push lambda
                ); //...estimate normals using FLANN

                // Add a button for orienting normals using FLANN
                viewer.ngui->addButton(
                        /* Displayed label: */ "Orient normals (FLANN)",

                        /* Lambda to call: */ [&]() {
                            // Store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // Check, if normals already exist
                            if (!cloud.hasNormals())
                                cloud.setNormals(
                                        acq::recalcNormals(
                                                /* [in]      K-neighbours for FLANN: */ kNeighbours,
                                                /* [in]             Vertices matrix: */ cloud.getVertices(),
                                                /* [in]      max neighbour distance: */ maxNeighbourDist
                                        )
                                );

                            // Estimate neighbours using FLANN
                            acq::NeighboursT const neighbours =
                                    acq::calculateCloudNeighbours(
                                            /* [in]        Cloud: */ cloud.getVertices(),
                                            /* [in] k-neighbours: */ kNeighbours,
                                            /* [in]      maxDist: */ maxNeighbourDist
                                    );

                            // Orient normals in place using established neighbourhood
                            int nFlips =
                                    acq::orientCloudNormals(
                                            /* [in    ] Lists of neighbours: */ neighbours,
                                            /* [in,out]   Normals to change: */ cloud.getNormals()
                                    );
                            std::cout << "nFlips: " << nFlips << "/" << cloud.getNormals().size() << "\n";

                            // Update viewer
                            acq::setViewerNormals(
                                    /* [in, out] Viewer to update: */ viewer,
                                    /* [in]            Pointcloud: */ cloud.getVertices(),
                                    /* [in] Normals of Pointcloud: */ cloud.getNormals()
                            );
                        } //...lambda to call on buttonclick
                ); //...addButton(orientFLANN)


                // Add new group
                viewer.ngui->addGroup("Connectivity from faces ");

                // Add a button for estimating normals using faces as neighbourhood
                viewer.ngui->addButton(
                        /* Displayed label: */ "Estimate normals (from faces)",

                        /* Lambda to call: */ [&]() {
                            // Store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // Check, if normals already exist
                            if (!cloud.hasNormals())
                                cloud.setNormals(
                                        acq::recalcNormals(
                                                /* [in]      K-neighbours for FLANN: */ kNeighbours,
                                                /* [in]             Vertices matrix: */ cloud.getVertices(),
                                                /* [in]      max neighbour distance: */ maxNeighbourDist
                                        )
                                );

                            // Estimate neighbours using FLANN
                            acq::NeighboursT const neighbours =
                                    acq::calculateCloudNeighboursFromFaces(
                                            /* [in] Faces: */ cloud.getFaces()
                                    );

                            // Estimate normals for points in cloud vertices
                            cloud.setNormals(
                                    acq::calculateCloudNormals(
                                            /* [in]               Cloud: */ cloud.getVertices(),
                                            /* [in] Lists of neighbours: */ neighbours
                                    )
                            );

                            // Update viewer
                            acq::setViewerNormals(
                                    /* [in, out] Viewer to update: */ viewer,
                                    /* [in]            Pointcloud: */ cloud.getVertices(),
                                    /* [in] Normals of Pointcloud: */ cloud.getNormals()
                            );
                        } //...button push lambda
                ); //...estimate normals from faces

                // Add a button for orienting normals using face information
                viewer.ngui->addButton(
                        /* Displayed label: */ "Orient normals (from faces)",

                        /* Lambda to call: */ [&]() {
                            // Store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // Check, if normals already exist
                            if (!cloud.hasNormals())
                                cloud.setNormals(
                                        acq::recalcNormals(
                                                /* [in]      K-neighbours for FLANN: */ kNeighbours,
                                                /* [in]             Vertices matrix: */ cloud.getVertices(),
                                                /* [in]      max neighbour distance: */ maxNeighbourDist
                                        )
                                );

                            // Orient normals in place using established neighbourhood
                            int nFlips =
                                    acq::orientCloudNormalsFromFaces(
                                            /* [in    ] Lists of neighbours: */ cloud.getFaces(),
                                            /* [in,out]   Normals to change: */ cloud.getNormals()
                                    );
                            std::cout << "nFlips: " << nFlips << "/" << cloud.getNormals().size() << "\n";

                            // Update viewer
                            acq::setViewerNormals(
                                    /* [in, out] Viewer to update: */ viewer,
                                    /* [in]            Pointcloud: */ cloud.getVertices(),
                                    /* [in] Normals of Pointcloud: */ cloud.getNormals()
                            );
                        } //...lambda to call on buttonclick
                ); //...addButton(orientFromFaces)


                // Add new group
                viewer.ngui->addGroup("Util");

                // Add a button for flipping normals
                viewer.ngui->addButton(
                        /* Displayed label: */ "Flip normals",
                        /*  Lambda to call: */ [&](){
                            // Store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // Flip normals
                            cloud.getNormals() *= -1.f;

                            // Update viewer
                            acq::setViewerNormals(
                                    /* [in, out] Viewer to update: */ viewer,
                                    /* [in]            Pointcloud: */ cloud.getVertices(),
                                    /* [in] Normals of Pointcloud: */ cloud.getNormals()
                            );
                        } //...lambda to call on buttonclick
                );

                // Add a button for setting estimated normals for shading
                viewer.ngui->addButton(
                        /* Displayed label: */ "Set shading normals",
                        /*  Lambda to call: */ [&](){

                            // Store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // Set normals to be used by viewer
                            viewer.data.set_normals(cloud.getNormals());

                        } //...lambda to call on buttonclick
                );

                // ------------------------
                // Dummy libIGL/nanoGUI API demo stuff:
                // ------------------------

                // Add new group
                viewer.ngui->addGroup("Dummy GUI demo");

                // Expose variable directly ...
                viewer.ngui->addVariable("float", floatVariable);

                // ... or using a custom callback
                viewer.ngui->addVariable<bool>(
                        "bool",
                        [&](bool val) {
                            boolVariable = val; // set
                        },
                        [&]() {
                            return boolVariable; // get
                        }
                );

                // Expose an enumaration type
                viewer.ngui->addVariable<Orientation>("Direction",dir)->setItems(
                        {"Up","Down","Left","Right"}
                );

                // Add a button
                viewer.ngui->addButton(
                        /* Displayed label: */ "Non-Rigid ICP",
                        /*  Lambda to call: */ [&](){
                            mesh msh;
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(1);
                            MatrixXi Temp_F = cloud.getFaces();
                            MatrixXd Temp_V = cloud.getVertices();
                            acq::DecoratedCloud &cloud2 = cloudManager.getCloud(2);
                            MatrixXi Target_F = cloud2.getFaces();
                            MatrixXd Target_V = cloud2.getVertices();
//                            SparseMatrix<double> A = msh.Adjacency_Matrix(Temp_F);
//                            SparseMatrix<double> M = msh.Incidence_Matrix(A);
                            MatrixXd new_V = msh.non_rigid_ICP(Temp_V, Temp_F, Target_V, Target_F);
                            cout << new_V.row(0) << endl;
                            cout << new_V.row(100) << endl;
                            cout << new_V.row(200) << endl;
                            //cout << new_V;
                            //cout<< A;
                            //Eigen::MatrixXd i = msh.non_rigid_ICP(V, F);
                            cloudManager.setCloud(acq::DecoratedCloud(new_V, Temp_F),0);

                            viewer.data.clear();
                            // Show mesh
                            viewer.data.set_mesh(cloudManager.getCloud(0).getVertices(),cloudManager.getCloud(0).getFaces());
                        });


                // Generate menu
                viewer.screen->performLayout();

                return false;
            }; //...viewer menu


    // Start viewer
    viewer.launch();

    return 0;
} //...main()
