#include <iostream>
#include <Eigen/Dense>
#include <sstream>
#include <string>
#include <iomanip>
#include <vector>

#ifdef DEBUG
    #include "Config.h"
#endif

#define SWAP 'S'
#define MUL 'M'
#define ADD 'A'
#define SOLUTION "SOLUTION"
#define DEGENERATE "DEGENERATE"
#define PRINT "PRINT"

class Point3D{
public:
    double x;
    double y;
    double z;
    double w = 1.0;
    
    Point3D(double _x, double _y, double _z): x(_x), y(_y), z(_z){};
};

class Point2D{
public:
    double x;
    double y;
    double depth;

    Point2D(double _x, double _y, double _depth): x(_x), y(_y), depth(_depth){};
};

class Image_plane_size{
public:
    double width;
    double height;
    Image_plane_size(double w, double h): width(w), height(h){}
};

class Focal{
public:
    double x;
    double y;
    Focal(double _x, double _y): x(_x), y(_y){}
};

class Center{
public:
    double x;
    double y;
    Center(double _x, double _y): x(_x), y(_y){}
};


class Intrinsic_params{
public:
    Focal _focal;
    Center _center;
    Intrinsic_params(double f_x, double f_y, double c_x, double c_y): _focal(f_x, f_y), _center(c_x, c_y){}
};

class FOV{
public:
    Eigen::MatrixXd intrinsic;
    Image_plane_size _image_plane_size;
    Intrinsic_params _intrinsic_params;
    double omega;

    FOV(double width, double height, double focal_x, double focal_y, double center_x, double center_y, double _omega): _image_plane_size(width, height), _intrinsic_params(focal_x, focal_y, center_x, center_y), omega(_omega){}
};

class Pinehole{
public:
    Eigen::MatrixXd mat_intrinsic;
    Image_plane_size _image_plane_size;
    Intrinsic_params _intrinsic_params;

    Pinehole(double width, double height, double focal_x, double focal_y, double center_x, double center_y, double _omega): _image_plane_size(width, height), _intrinsic_params(focal_x, focal_y, center_x, center_y){}

};


template <class T>
std::vector<T> input_manager(){
    std::vector<T> cameras;

    // input
    /*
    1: camera model for the first camera
    2: camera model for the second camera
    pinehole <width> <height> <focal x> <focal y> <center x> <center y>
    fov <width> <height> <focal x> <focal y> <center x> <center y> <omega>
    */

    std::string camera_model;
    std::cin>>camera_model;







    return cameras;
}


int main(int argc, char const *argv[])
{
    std::cout<<std::setprecision(20);

    //given
    /*
    - location of a point in the first image(green)
    - the depth, rotation and translation
    - then two camera calibration (two type of camera model: Pine-hole and FOV)
    */

    #if DEBUG == ON
    std::cout<<"Test"<<std::endl;
    #endif

    return 0;
}
