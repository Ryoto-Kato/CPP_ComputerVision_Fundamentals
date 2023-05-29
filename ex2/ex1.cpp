#include <iostream>
#include <Eigen/Dense>
#include <sstream>
#include <string>
#include <iomanip>
#include <vector>
#include <memory>

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
    double w;
    
    Point3D(double _x, double _y, double _z, double _w): x(_x), y(_y), z(_z), w(_w){};
    Point3D(std::vector<double> & vec): x(vec[0]), y(vec[1]), z(vec[2]), w(vec[3]){};
    ~Point3D(){}

    void print(){
        std::cout<<"x,y,z,w ="<<x<<","<<y<<","<<z<<","<<w<<std::endl;
    }

    double norm(){
        double _norm = std::sqrt((x*x+y*y)/(z*z));
        return _norm;
    }

};

class Point2D{
public:
    double x;
    double y;
    double depth;

    Point2D(double _x, double _y, double _depth): x(_x), y(_y), depth(_depth){};
    Point2D(std::vector<double> & vec): x(vec[0]), y(vec[1]), depth(vec[2]){};
    ~Point2D(){}

    void print(){
        std::cout<<"u, v, depth ="<<x<<","<<y<<","<<depth<<std::endl;
    }

    double norm(){
        // This should be applied after we devide the x and y by z
        double _norm = std::sqrt((x*x+y*y));
        return _norm;
    }
};

void input_line2mat(std::vector<Point2D> & points_f1){
    std::vector<double> line;
    double intermediate;

    while(std::cin>>intermediate){
        line.push_back(intermediate);
    }

    int counter = 0;
    std::vector<double> copy;

    for(unsigned int i = 0; i<line.size(); i++){
        copy.push_back(line[i]);
        if((i+1)%3 == 0){
            Point2D _point_f1 = {copy[0], copy[1], copy[2]};
            // input pixel coordinate (u, v) is a generic coordinate,
            // hence we need to multiply with depth to
            std::cout<<"input point on camera 1 image plane"<<std::endl;
            _point_f1.print();
            // obtain non-generic coordinate on image plane of cam2
            points_f1.push_back(_point_f1);
            copy.erase(copy.begin(), copy.end());
        }
    }
}


class Image_plane_size{
public:
    double width;
    double height;
    double pixel_ratio;
    Image_plane_size(double w, double h): width(w), height(h){
        pixel_ratio = width/height;
    }
    ~Image_plane_size(){}

};

class Focal{
public:
    double x;
    double y;
    Focal(double _x, double _y): x(_x), y(_y){}
    ~Focal(){}
};

class Center{
public:
    double x;
    double y;
    Center(double _x, double _y): x(_x), y(_y){}
    ~Center(){}

};


class Intrinsic_params{
public:
    Focal _focal;
    Center _center;
    Intrinsic_params(double f_x, double f_y, double c_x, double c_y): _focal(f_x, f_y), _center(c_x, c_y){}
    ~Intrinsic_params(){}
};

class Camera
{
public:
    std::string model;
    int id;
    int num_params;
    double omega;
    Eigen::MatrixXd intrinsic;
    Image_plane_size _image_plane_size;
    Intrinsic_params _intrinsic_params;

    Camera(double width, double height, double focal_x, double focal_y, double center_x, double center_y, double _omega): _image_plane_size(width, height), _intrinsic_params(focal_x, focal_y, center_x, center_y), omega(_omega){}
    Camera(int _id, std::string _model, std::vector<double> & _params): id(_id), model(_model), _image_plane_size(_params[0], _params[1]), _intrinsic_params(_params[2], _params[3], _params[4], _params[5]){
        if(_model == "fov"){
            omega = _params[6];
            num_params = 7;
        }else{
            omega = -1;
            num_params = 6;
        }

        intrinsic = Eigen::MatrixXd::Zero(3, 3);

        //inserting parameters to the intrinsic matrix

        intrinsic(0,0) = _intrinsic_params._focal.x;
        intrinsic(1,1) = _intrinsic_params._focal.y;
        intrinsic(2,2) = 1.0;
        intrinsic(0, 2) = _intrinsic_params._center.x;
        intrinsic(1, 2) = _intrinsic_params._center.y;

    }

    void print_params(){
        std::cout<<"--------------"<<model<<"----------------"<<std::endl;
        std::cout<<"Parameters"<<std::endl;

        std::cout<<"Image plane size: "<<std::endl;
        std::cout<<"(width, height, ratio) = "<<"("<<_image_plane_size.width<<","<<_image_plane_size.height<<","<<_image_plane_size.pixel_ratio<<")"<<std::endl;
        std::cout<<""<<std::endl;
        std::cout<<"Intrinsic Parameters"<<std::endl;
        std::cout<<"focal_x: "<<_intrinsic_params._focal.x<<std::endl;
        std::cout<<"focal_y: "<<_intrinsic_params._focal.y<<std::endl;
        std::cout<<"center_x: "<<_intrinsic_params._center.x<<std::endl;
        std::cout<<"center_y: "<<_intrinsic_params._center.y<<std::endl;
        std::cout<<"omega: "<<omega<<std::endl;
        std::cout<<std::endl;

        std::cout<<"intrinsic matrix"<<std::endl;
        std::cout<<intrinsic<<std::endl;

        std::cout<<"----------------------------------------"<<std::endl;
    }
};

double inline ptam(Point2D & coord_2d, double omega){
    long double result = 0.0;
    double _norm = coord_2d.norm();
    result = (1/(omega*_norm))*atan(2*_norm*tan(omega/2));
    std::cout<<"distortion coefficient"<<std::endl;
    std::cout<<result<<std::endl;
    return result;
}

Point3D backprojection(Point2D & _2d, const Camera * cam){
    double c_x = cam->_intrinsic_params._center.x;
    double c_y = cam->_intrinsic_params._center.y;
    double f_x = cam->_intrinsic_params._focal.x;
    double f_y = cam->_intrinsic_params._focal.y;
    double width = cam->_image_plane_size.width;
    double height = cam->_image_plane_size.height;
    double pix_ratio = cam->_image_plane_size.pixel_ratio;

    Point3D _backprojected_point = {0, 0, 0, 1.0};

    if(cam->model == "fov"){
        // When we back-project the pixel coordinate
        // We do not need to consider the distortion coefficien
        // _2d.x /= _dist_coeff;
        // _2d.y /= _dist_coeff;

        Point3D _3d = {(_2d.x - c_x)/f_x, (_2d.y - c_y)/f_y, _2d.depth, 1.0};
        Point2D _centralized_2d = {_3d.x, _3d.y, _2d.depth};
        double _instance_norm = _centralized_2d.norm();

        double _omega = cam->omega;

        double _norm = (tan(_omega * _instance_norm))/(2.0*tan(_omega/2.0));
        double _ptam = (1.0/(_omega*_norm))*atan(2.0*_norm*tan(_omega/2.0));
        _3d.x /= _ptam;
        _3d.y /= _ptam;
        _3d.x *=_2d.depth;
        _3d.y *=_2d.depth;
        // Point3D _3d = {(_2d.x - c_x)*_2d.depth/f_x*_dist_coeff, (_2d.y - c_y)*_2d.depth/f_y*_dist_coeff, _2d.depth, 1.0};
        _backprojected_point = _3d;
    }else{
        Point3D _3d = {(_2d.x - c_x)/f_x, (_2d.y - c_y)/f_y, _2d.depth, 1.0};
        _3d.x *=_2d.depth;
        _3d.y *=_2d.depth;
        _backprojected_point = _3d;
    }

    return _backprojected_point;
}

void round_1decimal_place(Eigen::MatrixXd & a){
    for(Eigen::Index i =0; i<a.rows(); i++){
        for(Eigen::Index j=0; j<a.cols(); j++){
            a(i, j) = (int)(a(i, j) * 1000 + 0.5);
            a(i, j) = (float)(a(i,j)/1000);
        }
    }
}

std::vector<Point2D> transform_c1_c2(std::vector<Point3D> & coords_list, Eigen::MatrixXd & rt_mat, const Camera * cam2){
    /*
    Transform coordinates of point from the camera 1 space to camera 2 space
    */
    Eigen::Index row = 4;
    Eigen::Index col = coords_list.size();

    Eigen::MatrixXd rotation_mat = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd translation_mat = Eigen::MatrixXd::Identity(4, 4);
    
    rotation_mat.block(0, 0, 3, 3) = rt_mat.block(0,0,3,3);

    translation_mat.block(0, 3, 3, 1) = rt_mat.block(0,3,3,1);

    std::cout<<"rotation matrix"<<std::endl;
    std::cout<<rotation_mat<<std::endl;

    std::cout<<"translation matrix"<<std::endl;
    std::cout<<translation_mat<<std::endl;
    //points matrix
    //no need to be transposed
    Eigen::MatrixXd points_in_c1(row, col); //shape = [4, col]

    for(unsigned int i = 0; i<coords_list.size(); i++){
            points_in_c1(0,i) = coords_list[i].x;
            points_in_c1(1,i) = coords_list[i].y;
            points_in_c1(2,i) = coords_list[i].z;
            points_in_c1(3,i) = 1.0;
    }

    std::cout<<"3Dpoint matrix"<<std::endl;
    std::cout<<points_in_c1<<std::endl;
    //Transform the matrix by multiplying the transformation matrix to the point matrix in camera 1 space

    Eigen::MatrixXd points_in_c2(4, col);

    std::cout<<"Transformation matrix"<<std::endl;
    std::cout<<rt_mat<<std::endl;


    // transform the cam1 coordinate to cam2 coordinate
    points_in_c2 = rotation_mat * translation_mat * points_in_c1;
    std::cout<<"points in camera 2 space"<<std::endl;
    std::cout<<points_in_c2<<std::endl;
    // We need to devide x and y by depth as the pinhole model
    // x_i = f*x_c/z_c from x_i/f = x_c/z_c

    std::vector<Point2D> pixel_coords_on_im2;

    double c_x = cam2->_intrinsic_params._center.x;
    double c_y = cam2->_intrinsic_params._center.y;

    for(int p = 0; p<points_in_c2.cols(); p++){
        //x
        points_in_c2(0, p) /=  points_in_c2(2, p);
        //y
        points_in_c2(1, p) /=  points_in_c2(2, p);
        Point2D _centralized_point = {points_in_c2(0, p), points_in_c2(1, p), points_in_c2(2, p)};        
        Point2D _point = {0.0, 0.0, 0.0};

        _point.x = points_in_c2(0, p);
        _point.y = points_in_c2(1, p);
        _point.depth = points_in_c2(2, p);

        if(cam2->model == "fov"){
            double _distortion_coeff = ptam(_centralized_point, cam2->omega);
            _point.x = _distortion_coeff * (_point.x);
            _point.y = _distortion_coeff * (_point.y);
        }

        _point.x = _point.x * cam2->_intrinsic_params._focal.x + cam2->_intrinsic_params._center.x;
        _point.y = _point.y * cam2->_intrinsic_params._focal.y + cam2->_intrinsic_params._center.y;
        pixel_coords_on_im2.push_back(_point);
        // pixel_coords_on_im2.push_back(_point);
        // if(cam2->model == "fov"){
        //     points_in_c2(0, p) *= ptam(_p, cam2->omega);
        //     points_in_c2(1, p) *= ptam(_p, cam2->omega);
        // }
    }

    // std::cout<<"After apply, distortion coeff, points in camera 2 space"<<std::endl;
    // std::cout<<points_in_c2<<std::endl;

    // Eigen::MatrixXd points_on_im2(3, col);

    // // Points cam2 pixel coordinate by applying intrinsic
    // points_on_im2 = cam2->intrinsic * points_in_c2;

    // std::cout<<"point on cam2 image plane"<<std::endl;
    // std::cout<<points_on_im2<<std::endl;

    // for(unsigned int i = 0; i<points_on_im2.cols(); i++){
    //     // points_on_im2(0,i)/=points_on_im2(2, i);
    //     // points_on_im2(1,i)/=points_on_im2(2, i);
    //     Point2D _p = {points_on_im2(0, i)-c_x, points_on_im2(1, i)-c_y, points_on_im2(2, i)};
    //     double distortion_coeff = 1.0;
    //     if(cam2->model == "fov"){
    //         distortion_coeff= ptam(_p, cam2->omega);
    //     }
    //     Point2D _point = {distortion_coeff*(points_on_im2(0, i)-c_x)+c_x, distortion_coeff*(points_on_im2(1, i)-c_y)+c_y, points_on_im2(2, i)};
    //     // if(_point.x >= 0 && _point.y >= 0 && _point.x <= cam2->_image_plane_size.width && _point.y <= cam2->_image_plane_size.height){
    //         pixel_coords_on_im2.push_back(_point);
    //     // }
    // }

    // std::cout<<"point pixel coordinate on cam2"<<std::endl;
    // for(auto & p : pixel_coords_on_im2){
    //     p.print();
    // }

    return pixel_coords_on_im2;
}

int main(int argc, char const *argv[])
{
    std::cout<<std::setprecision(6);

    //given
    /*
    - location of a point in the first image(green)
    - the depth, rotation and translation
    - then two camera calibration (two type of camera model: Pine-hole and FOV)
    */
    std::vector<Camera> cameras;
    std::string camera_model;

    // take inputs for camera configurations
    for(int i =0; i<2; i++){
        std::cout<<"Input camera"<<i<<" parameters...."<<std::endl;
        std::vector<double> camera_params;
        std::cin>>camera_model;
        double copy = 0.0;
        int _id = i;
        int _num_params = 0;

        if(camera_model == "fov"){
            _num_params = 7;
        }else{
            _num_params = 6;
        }
        
        for(int i = 0; i<_num_params; i++){
            std::cin>>copy;
            camera_params.push_back(copy);
        }

        Camera cam = {_id, camera_model,camera_params};
        cameras.push_back(cam);
        cameras[i].print_params();
    }

    // take rotation and translation
    Eigen::MatrixXd transformation_mat = Eigen::MatrixXd::Zero(3, 4);
    double copy;

    for(unsigned int i = 0; i < 3; i++){
        for(unsigned int j = 0; j < 4; j++){
            std::cin>>copy;
            transformation_mat(i, j)= copy;
        }
    }

    std::cout<<"rotation matrix"<<std::endl;
    std::cout<<transformation_mat<<std::endl;

    std::vector<Point2D> coords_on_im1;

    input_line2mat(coords_on_im1);

    std::vector<Point2D> generic_coords_on_im1 = coords_on_im1;
    Eigen::MatrixXd generic_im1_points = Eigen::MatrixXd::Zero(3, generic_coords_on_im1.size());

    int index =0;
    for(auto & p : generic_coords_on_im1){
        p.depth = 1.0;
        if(cameras[0].model == "fov"){
            double _dist_coeff = ptam(coords_on_im1[index], cameras[0].omega);
            p.x /= _dist_coeff;
            p.y /= _dist_coeff;
        }
        index++;
    }

    Eigen::MatrixXd inverse_cam1_intrinsic = cameras[0].intrinsic.inverse();
    //coordinates in camera 1 space

    for(unsigned int i = 0; i<generic_coords_on_im1.size(); i++){
            generic_im1_points(0,i) = generic_coords_on_im1[i].x;
            generic_im1_points(1,i) = generic_coords_on_im1[i].y;
            generic_im1_points(2,i) = generic_coords_on_im1[i].depth;
    }

    Eigen::MatrixXd cam1_mat_points = inverse_cam1_intrinsic *  generic_im1_points;
    std::vector<Point2D> coords_in_cam1;

    Eigen::MatrixXd cam2_mat_points = 

    std::cout<<"coordinates in cam1"<<std::endl;
    for(unsigned int num_p = 0; num_p < cam1_mat_points.cols(); num_p++){
        Point2D _p = {cam1_mat_points(0, num_p), cam1_mat_points(1, num_p), cam1_mat_points(2, num_p)};
        _p.print();
        coords_in_cam1.push_back(_p);
    }

    std::cout<<"Points in camera 1 coordinate"<<std::endl;
    std::cout<<cam1_mat_points<<std::endl;

    

    

    // int counter = 0;
    // for(auto & a : coords_on_im1){
    //     a.print();
    //     std::cout<<"Back-projection: ";
    //     Point3D _backprojected_coords = backprojection(a, &cameras[0]);
    //     coords_in_c1.push_back(_backprojected_coords);
    //     coords_in_c1[counter].print();
    //     counter++;
    // }

    //convert
    // std::vector<Point2D> result = transform_c1_c2(coords_in_c1, transformation_mat, &cameras[1]);



    // for(unsigned int i = 0; i<result.size(); i++){
    //     //if(result[i].depth >= 0 && result[i].x >= 0 && result[i].y >= 0 && result[i].x <= cameras[1]._image_plane_size.width && result[i].y <= cameras[1]._image_plane_size.height){
    //         result[i].print();
    //     //}else{
    //     //    std::cout<<"OB"<<std::endl;
    //     //}
    // }

    return 0;
}
