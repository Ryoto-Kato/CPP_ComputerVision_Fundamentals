#include <iostream>
#include <Eigen/Dense>
#include <sstream>
#include <string>
#include <iomanip>
#include <vector>
#include <memory>
#include <iterator>
#include <map>

#define DEBUG true
typedef std::vector<Eigen::MatrixXd> Mats; 

auto print_vect=[](auto const & v){
    std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout<<std::endl;
};

void comb_01(std::vector<int> & sets, int len, std::vector<int> & a, int n, std::vector<std::vector<int>> & output){

    //basecase
    //n = 0

    if(n==0){
        output.push_back(a);
        return;
    }

    /*
    i=0
    n=2
`   newPrefix = 0
                    i = 0
                    n = 1
                    newPrefix = 00
                                    i = 0
                                    n = 0
                                    newPrefix = 000
                                                    output<<000
                                                    return
                                    i = 1
                                    n = 0
                                    newPrefix = 001
                                    output<<001
                                    
                                    return
                    i = 1
                    n = 1
                    newPrefix = 01
                                    i= 0
                                    n = 0
                                    newPrefix = 010
                                                        output<<010
                                                        return
                                    i= 1
                                    n= 1
                                    newPrefix = 011
                                                        output<<011
                                                        return
    i = 1
    n = 2
    newPrefix = 1
                    i = 0
                    newPrefix = 10
                                    i = 0
                                    newPrefix = 100
                                                        output<<100
                                                        return
                                    i = 1
                                    newPrefix = 101
                                    output<<101
                                    return
                    i = 1
                    newPrefix = 11
                                    i = 0
                                    newPrefix = 110
                                                        output<<110
                                                        return
                                    i = 1
                                    newPrefix = 111
                                                        output<<111
                                                        return
            return
    return
    */
    for(unsigned int i = 0; i<len; i++){

        std::vector<int> newPrefix;

        newPrefix.assign(a.begin(), a.end());
        newPrefix.push_back(sets[i]);

        //n is decreased
        comb_01(sets, len, newPrefix, n-1, output);        
    }

}

class Point3D{
public:
    double x;
    double y;
    double z;
    double w;
    
    Point3D(double _x, double _y, double _z, double _w): x(_x), y(_y), z(_z), w(_w){};
    Point3D(std::vector<int> & vec): x(vec[0]), y(vec[1]), z(vec[2]), w(vec[3]){};
    Point3D(std::vector<double> & vec): x(vec[0]), y(vec[1]), z(vec[2]), w(vec[3]){};
    ~Point3D(){}

    void print(){
        std::cout<<"x,y,z,w ="<<x<<","<<y<<","<<z<<","<<w<<std::endl;
    }

    Eigen::Vector4d _vector(){
        Eigen::Vector4d _a = {x, y, z, w};
        return _a;
    }

    Eigen::Vector4d scaled_by(double s){
        Eigen::Vector4d _original = _vector();
        return s*_original;
    }

    double norm(){
        double _norm = std::sqrt((x*x+y*y+z*z));
        return _norm;
    }
};

std::vector<int> merge(const std::vector<int> & left, const std::vector<int> & right){
    std::vector<int> merged_vector;
    std::merge(left.begin(), left.end(), right.begin(), right.end(), std::back_inserter(merged_vector));

    return merged_vector;
}

std::vector<Point3D> cube_config(const int & num_edge, bool _print){
        std::vector<Point3D> unit_cube;
        std::vector<std::vector<int>> comb01_lists;
        std::vector<int> sets = {0,1};
        std::vector<int> a;
        int dim = 3;
        comb_01(sets, sets.size(), a, dim, comb01_lists);

        for(unsigned int i = 0; i<comb01_lists.size(); i++){
            comb01_lists[i].push_back(1.0);
            Point3D _vertex = {comb01_lists[i]};
            unit_cube.push_back(_vertex);
            if(_print){
                unit_cube[i].print();                
            }
        }

        return unit_cube;
}

class Point2D{
public:
    double x;
    double y;
    double depth;

    Point2D(double _x, double _y, double _depth): x(_x), y(_y), depth(_depth){};
    Point2D(std::vector<double> & vec): x(vec[0]), y(vec[1]), depth(vec[2]){};
    ~Point2D(){}

    void print(){
        std::cout<<std::right<<"x,y,depth:\t"<<x<<","<<y<<","<<depth<<std::endl;
    }

    Eigen::Vector3d eigen_vector(){
        Eigen::Vector3d _a = {x, y, depth};
        return _a;
    }

    double norm(){
        // This should be applied after we devide the x and y by z
        double _norm = std::sqrt((x*x+y*y));
        return _norm;
    }

    Eigen::MatrixXd skew_symmetric_mat(){
        Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(3, 3);
        mat << 0, (-1)*depth, y,
            depth, 0, (-1)*x,
            (-1)*y, x, 0;

        return mat;
    }
};

void pixcoords_input_reader(std::vector<Point2D> & points_f1, bool _print){
    std::vector<double> line;
    double intermediate;

    while(std::cin>>intermediate){
        line.push_back(intermediate);
    }

    int counter = 0;
    std::vector<double> copy;
    for(unsigned int i = 0; i<line.size(); i++){
        copy.push_back(line[i]);
        if((i+1)%2 == 0){
            Point2D _point_f1 = {copy};            
            points_f1.push_back(_point_f1);
            copy.erase(copy.begin(), copy.end());
            if(_print){
                std::cout<<"2d pixel coords"<<"_"<<counter<<std::endl;
                _point_f1.print();
            }
            counter++;
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
    unsigned int num_pix_coords = -1;
    Eigen::MatrixXd intrinsic;
    Image_plane_size _image_plane_size;
    Intrinsic_params _intrinsic_params;
    std::vector<Point2D> pixel_coords;
    std::vector<Point2D> normalized_coords;

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

    void get_noramlized_coords(){
        double * f_x = &(_intrinsic_params._focal.x);
        double * f_y = &(_intrinsic_params._focal.y);
        double * c_x = &(_intrinsic_params._center.x);
        double * c_y = &(_intrinsic_params._center.y);

        for(auto & item : pixel_coords){
            double * _u = &(item.x);
            double * _v = &(item.y);
            if((*_u) == -1 && (*_v) == -1){
                normalized_coords.push_back(item);
            }else{
                Point2D normalized_point = {((*_u)-(*c_x))/(*f_x), ((*_v)-(*c_y))/(*f_y), item.depth};
                normalized_coords.push_back(normalized_point);
            }
        }
    }

    void print_params(){
        std::cout<<"--------------"<<model<<"_"<<id<<"----------------"<<std::endl;
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

    void print_pixelcoords(bool is_normalized){
        std::string title = is_normalized? "Normalized" : "Pixel";
        std::cout<<title<<" coordinate"<<"camera"<<id<<std::endl;
        std::vector<Point2D> * pointer_to_coordinates;
        if(is_normalized){
            pointer_to_coordinates = &(normalized_coords);
        }else{
            pointer_to_coordinates = &(pixel_coords);
        }
        for(auto & item : *pointer_to_coordinates){
            item.print();
        }
    }

    Eigen::MatrixXd getKRT(Eigen::MatrixXd & _SE){
        return intrinsic * _SE;
    }
};


std::vector<bool> flag_corresp(Camera & cam1, Camera & cam2, bool _debug){
    std::vector<bool> flags;
    std::cout<<"Num of pix coords: "<<cam1.num_pix_coords<<std::endl;
    for(unsigned int s = 0; s<cam1.num_pix_coords; s++){
        bool _f = true;
        if((cam1.normalized_coords[s].x == -1 && cam1.normalized_coords[s].y == -1) || (cam2.normalized_coords[s].x == -1 && cam2.normalized_coords[s].y == -1)){
            _f = false;
        }
        flags.push_back(_f);
    }

    if(_debug){
        std::cout<<"Flags coorespondences"<<std::endl;
        for(unsigned int i = 0; i<flags.size(); i++){
            std::cout<<flags[i]<<std::endl;
        }
    }

    return flags;
}

double inline ptam(Point2D & coord_2d, double omega){
    long double result = 0.0;
    double _norm = coord_2d.norm();
    result = (1/(omega*_norm))*atan(2*_norm*tan(omega/2));
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

        //_3d.x = x_c * g(r) / z_c
        //_3d.y = y_c * g(r) / z_c
        Point3D _3d = {(_2d.x - c_x)/f_x, (_2d.y - c_y)/f_y, 1.0, 1.0};
        Point2D _centralized_2d = {_3d.x, _3d.y, _2d.depth};
        double _instance_norm = _centralized_2d.norm();
        double _r = ((0.5)*tan(cam->omega * _instance_norm))/tan(cam->omega/2.0);
        double _ptam = (1/(cam->omega *_r))*atan(2*_r*tan(cam->omega /2));

        _3d.x /= _ptam;
        _3d.y /= _ptam;

        double _3d_point_norm = _3d.norm();

        //_3d.x, _3d.y, _3d.z = x_c, y_c. z_c
        _3d.z = _2d.depth/_3d_point_norm;
        _3d.x *= _3d.z;
        _3d.y *= _3d.z;

        // Point3D _3d = {(_2d.x - c_x)*_2d.depth/f_x*_dist_coeff, (_2d.y - c_y)*_2d.depth/f_y*_dist_coeff, _2d.depth, 1.0};
        _backprojected_point = _3d;
    }else{
        //_3d.x = x_c / z_c
        //_3d.y = y_c / z_c
        Point3D _3d = {(_2d.x - c_x)/f_x, (_2d.y - c_y)/f_y, 1.0, 1.0};

        double _3d_point_norm = _3d.norm();

        _3d.z = _2d.depth/_3d_point_norm;
        _3d.x *=_3d.z;
        _3d.y *=_3d.z;

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

    //points matrix
    //no need to be transposed
    Eigen::MatrixXd points_in_c1(row, col); //shape = [4, col]

    for(unsigned int i = 0; i<coords_list.size(); i++){
            points_in_c1(0,i) = coords_list[i].x;
            points_in_c1(1,i) = coords_list[i].y;
            points_in_c1(2,i) = coords_list[i].z;
            points_in_c1(3,i) = 1.0;
    }

    Eigen::MatrixXd points_in_c2(4, col);

    points_in_c2 = rt_mat * points_in_c1;

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
    }

    return pixel_coords_on_im2;
}

void camera_calibration(std::vector<Point3D> & world_coords, std::vector<Point2D> & pixel_coords, bool _print){
    int num_vertex = world_coords.size();
    //Tsai's method
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(3, 4);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(2*num_vertex, 12);

    for(unsigned int i=0; i<num_vertex; i++){
        Eigen::MatrixXd dot_prod = Eigen::MatrixXd::Zero(2, 4);
        double _mu = (-1)*pixel_coords[i].x;
        double _mv = (-1)*pixel_coords[i].y;

        dot_prod.row(0) = world_coords[i].scaled_by(_mu);
        dot_prod.row(1) = world_coords[i].scaled_by(_mv);

        // std::cout<<dot_prod<<std::endl;

        Eigen::Index row1= 2*i;
        Eigen::Index row2 = 2*i+1;
        Eigen::Vector4d _row = world_coords[i]._vector();

        //concatenating the elements in dot_prod and world_coordinate to create Matrix Q
        Q.block(row1, 0, 1, 4) = _row.transpose();
        Q.block(row1, 8, 1, 4) = dot_prod.row(0);
        Q.block(row2, 4, 1, 4) = _row.transpose();
        Q.block(row2, 8, 1, 4) = dot_prod.row(1);

    }

    //Perform SVD with respect to the Q matrix
    //the eigen-vector corresponding to the smallest eigenvalue it the kernel of the matrix
    Eigen::MatrixXd System_mat = Q;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(System_mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

    svd.compute(System_mat);

    // Eigen::MatrixXd V_tranposed = svd.matrixV().transpose();
    Eigen::MatrixXd V = svd.matrixV();

    // std::cout<<V_tranposed<<std::endl;
    // std::cout<<V<<std::endl;
    // std::cout<<"Number of column"<<std::endl;
    // std::cout<<V.cols()<<std::endl;


    Eigen::VectorXd solution_m = V.col(V.cols()-1);

    // std::cout<<solution_m.norm()<<std::endl;
    //unstacking the solution m
    for(unsigned int i = 0; i<solution_m.size(); i++){
        unsigned int row_index = (int)(i/4);
        unsigned int col_index = (int)(i%4);

        M(row_index, col_index) = solution_m(i); 
    }

    //QR factorization to obtain the upper triangle matrix and the orthonormal matrix
    //R should be upper triangle matrix indicating the camera intrinsic matrix
    //Q is the unitary matrix which can encode R|T

    //only focus on KR recovering by QR decomposition
    //permutation matrix
    Eigen::MatrixXd _Perm = Eigen::MatrixXd::Zero(3, 3);
    _Perm(0, 2) = 1.0;
    _Perm(1, 1) = 1.0;
    _Perm(2, 0) = 1.0;

    // std::cout<<"Permutation matrix (reverse rows)"<<std::endl;
    // std::cout<<_Perm<<"\n"<<std::endl;

    Eigen::MatrixXd _deQ, _deR, _intrinsic, _rotation, _translation, _transformation;
    
    Eigen::MatrixXd square_M = M.block(0, 0, 3, 3);

    // std::cout<<"Original croped square matrix M"<<std::endl;
    // std::cout<<square_M<<"\n"<<std::endl;
    
    Eigen::MatrixXd Perm_sqM = _Perm * square_M;

    // std::cout<<"Permuted matrix square_M"<<std::endl;
    // std::cout<<Perm_sqM<<"\n"<<std::endl;

    // std::cout<<"Transposed Permuted matrx square M"<<std::endl;
    // std::cout<<Perm_sqM.transpose()<<"\n"<<std::endl;

    //apply QR decomposition to Perm_sqM.transpose()

    auto QR1 = Perm_sqM.transpose().householderQr();
    Eigen::MatrixXd _qr1_q = QR1.householderQ();
    Eigen::MatrixXd _qr1_r = QR1.matrixQR().triangularView<Eigen::Upper>();

    // std::cout<<"unitary matrix (QR1)"<<std::endl;
    // std::cout<<_qr1_q<<std::endl;

    // std::cout<<"upper triangle matrix (QR1)"<<std::endl;
    // std::cout<<_qr1_r<<std::endl;

    _rotation = _Perm * _qr1_q.transpose();
    _intrinsic = _Perm * _qr1_r.transpose() * _Perm;

    // focus on KT

    Eigen::MatrixXd M_partT = M.block(0, 3, 3, 1);
    
    _translation = _intrinsic.inverse() * M_partT;

    _transformation = Eigen::MatrixXd::Zero(3, 4);
    _transformation.block(0, 0, 3, 3) = _rotation;
    _transformation.block(0, 3, 3, 1) = _translation;

    if(_print){
        std::cout<<"Transformation matrix"<<std::endl;
        std::cout<<_transformation<<std::endl;

        std::cout<<"Intrinsic matrix"<<std::endl;
        std::cout<<_intrinsic<<std::endl;
    }

    if(_transformation(2, 3) < 0){
        _intrinsic.col(2) *= (-1);
        _transformation.row(2)*=(-1);
    }

    if(_print){
        std::cout<<"Transformation matrix"<<std::endl;
        std::cout<<_transformation<<std::endl;

        std::cout<<"Intrinsic matrix"<<std::endl;
        std::cout<<_intrinsic<<std::endl;
    }

    Eigen::Vector3d rows_negative;
    rows_negative << 0, 0, 0;
    Eigen::Matrix3d _intrinsic_mask;
    _intrinsic_mask << 1, 0, 1, 0, 1, 1, 0, 0, 1;

    Eigen::MatrixX3i _int_intrinsic = Eigen::MatrixX3i::Zero(3,3);

    for(unsigned int row = 0; row<_intrinsic.rows(); row++){
        for(unsigned int col = 0; col<_intrinsic.cols(); col++){
            if(_intrinsic_mask(row, col) == 1){
                if(_intrinsic(row, col) < 0){
                    rows_negative(row) = 1;
                    _intrinsic(row, col) *= (-1);
                }
            }

            if(row == 0 && col == 1){
                _intrinsic(row, col) = 0;
            }
        }
    }

    for(unsigned int _r = 0; _r<_intrinsic.rows(); _r++){
        if(rows_negative(_r) == 1){
            _transformation.row(_r)*=(-1);
        }
    }

    // std::cout<<"index of row to be flip its sign"<<std::endl;
    // std::cout<<rows_negative<<std::endl;

    double factor = abs(_intrinsic(2, 2));

    for(unsigned int i = 0; i<_intrinsic.rows(); i++){
        for(unsigned int j = 0; j<_intrinsic.cols(); j++){
            if(_intrinsic_mask(i, j) == 1){
                _intrinsic(i,j) /= factor;
                int times_ten = _intrinsic(i,j) * 10.0;

                if(times_ten%10 < 5){
                    // std::cout<<_intrinsic(i, j)<<std::endl;
                    _int_intrinsic(i,j)=floor(_intrinsic(i, j));   
                }else{
                    // std::cout<<_intrinsic(i, j)<<std::endl;
                    _int_intrinsic(i,j)=ceil(_intrinsic(i, j));  
                }
            }
        }
    }

    // std::cout<<"Final::Camera Intrinsic"<<std::endl;
    std::cout<<_int_intrinsic<<std::endl;

    std::cout<<std::endl;

    // std::cout<<"Final::Transformation matrix"<<std::endl;
    std::cout<<_transformation<<std::endl;

}

//ex4
void SetCamParams(std::vector<Camera> & cameras, unsigned int num_cam, bool _debug){
    std::string camera_model;
    // take inputs for camera configurations
    for(int i =0; i<num_cam; i++){
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
        if(_debug){
            cameras[i].print_params();
        }
    }
}

//ex4
void assign_pixelcoords2cameras(std::vector<Camera> & cameras, std::vector<Point2D> & pixel_coords, bool _debug){
    unsigned int counter=0;
    unsigned int nums_pix_cds = 0;

    for(auto & item: pixel_coords){
        item.depth = 1.0;
        cameras[counter].pixel_coords.push_back(item);   
        counter++;
        if(counter==3){
            counter = 0;
            nums_pix_cds++;
        }
    }

    cameras[0].num_pix_coords = nums_pix_cds;
    cameras[1].num_pix_coords = nums_pix_cds;
    cameras[2].num_pix_coords = nums_pix_cds;

    for(auto & cam: cameras){
        cam.get_noramlized_coords();
    }

    if(_debug){
        bool is_normalized = true;
        for(auto & cam : cameras){
            cam.print_pixelcoords(is_normalized);
        }
    }


}

//ex4
Eigen::MatrixXd kronocker_product(Camera & cam_1, Camera & cam_2, bool _debug){
    unsigned int num_corresp = cam_1.num_pix_coords;
    if(num_corresp!=cam_2.num_pix_coords){
        std::cout<<"ERROR: "<<"The number of pixels are not identical in two cameras"<<std::endl;
    }

    std::vector<bool> _vec_flags = flag_corresp(cam_1, cam_2, DEBUG);

    unsigned int actual_num_corresp = 0;
    for(unsigned int _id = 0; _id < _vec_flags.size(); _id++){
        if(_vec_flags[_id]){
            actual_num_corresp++;
        }
    }

    std::cout<<"Actual_num_corresp"<<actual_num_corresp<<std::endl;

    std::vector<Point2D>::iterator id_1 = cam_1.normalized_coords.begin();
    std::vector<Point2D>::iterator id_2 = cam_2.normalized_coords.begin();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ALPHA(8, 9);

    unsigned int _col = 0;
    for(unsigned int i_1 = 0; i_1<num_corresp; i_1++){
        bool _f = true;
        if(_col == ALPHA.cols()-1){
            break;
        }else{
            if(((*id_1).x == -1 && (*id_1).y == -1) || ((*id_2).x == -1 && (*id_2).y == -1)){
                _f = false;
            }

            if(_f){
                ALPHA(_col, 0) = (*id_2).x * (*id_1).x;
                ALPHA(_col, 1) = (*id_2).x * (*id_1).y;
                ALPHA(_col, 2) = (*id_2).x;
                ALPHA(_col, 3) = (*id_2).y * (*id_1).x;
                ALPHA(_col, 4) = (*id_2).y * (*id_1).y;
                ALPHA(_col, 5) = (*id_2).y;
                ALPHA(_col, 6) = (*id_1).x;
                ALPHA(_col, 7) = (*id_1).y;
                ALPHA(_col, 8) = (*id_1).depth;
                _col++;
            }
            id_1++;
            id_2++;
        }
    }

    if(_debug){
        std::cout<<"Given num_corresponds: "<<num_corresp<<std::endl;
        std::cout<<"ALPHA[num_corresp, 9] = shape: "<<ALPHA.rows()<<","<<ALPHA.cols()<<std::endl;
        std::cout<<ALPHA<<std::endl;
    }

    // Eigen::Map<Eigen::RowVectorXd> _vector_ALPHA(ALPHA.data(), ALPHA.size());
    // Eigen::VectorXd _verticalstacked_ALPHA = _vector_ALPHA.transpose();
    return ALPHA;
}

//row-major e_11, e_12, e_13, e_21, e_22, e_23, e_31, e_32, e_33
Eigen::MatrixXd map_Vect2Squaremat(Eigen::VectorXd & _vec){
    unsigned int dim = sqrt(_vec.size());
    std::cout<<dim<<"dimension"<<std::endl;
    Eigen::MatrixXd sq_mat = Eigen::MatrixXd::Zero(dim, dim);

    for(unsigned int i = 0; i<dim; i++){
        for(unsigned int j = 0; j<dim; j++){
            sq_mat(i, j) = _vec(dim*i+j);
        }
    }

    return sq_mat;
}

Eigen::Matrix3d rotaion_z(const double angle_z){
    Eigen::Matrix3d r_z;
    double eps = 1e-6;
    double cos_z = std::abs(std::cos(angle_z))<=eps? 0: std::cos(angle_z);
    double sin_z = std::abs(std::sin(angle_z))<=eps? 0: std::sin(angle_z);

    r_z << cos_z, sin_z, 0,
        (-1)*sin_z, cos_z, 0,
        0, 0, 1;
    return r_z;
}


std::pair<Mats,Mats> possible_EuclidianTrans(Eigen::JacobiSVD<Eigen::MatrixXd> & _svd, Eigen::MatrixXd & sigma, bool _debug){
    
    /*
    * ET4 has four possible solution for R and T
    * [3x3][3x3] in a single element x 4 possible solution
    * 0. R = U*Rz(+pi/2)*transV, ssm_T = U*Rz(+pi/2)sigma*U.transpose()
    * 1. R = U*Rz(+pi/2)*transV, ssm_T = U*Rz(-pi/2)sigma*U.transpose()
    * 2. R = U*Rz(-pi/2)*transV, ssm_T = U*Rz(+pi/2)sigma*U.transpose()
    * 3. R = U*Rz(-pi/2)*transV, ssm_T = U*Rz(-pi/2)sigma*U.transpose()
    */

    Eigen::MatrixXd U = _svd.matrixU();
    Eigen::MatrixXd transU = _svd.matrixU().transpose();
    Eigen::MatrixXd transV = _svd.matrixV();
    Eigen::Matrix3d r_z_plus = rotaion_z(0.5*M_PI);
    std::cout<<"r_z_plus"<<std::endl;
    std::cout<<r_z_plus<<std::endl;
    Eigen::Matrix3d r_z_minus = rotaion_z((-1)*0.5*M_PI);
    std::cout<<"r_z_minus"<<std::endl;
    std::cout<<r_z_minus<<std::endl;

    Eigen::Matrix3d decomposed_sigma;
    // decomposed_sigma << 0, 1, 0,
    //                 -1, 
    // std::cout<<r_z<<std::endl;

    //0. R = U*Rz(+pi/2)*transV, ssm_T = U*Rz(+pi/2)sigma*transV
    Mats stacked_R;
    Mats stacked_ssmT;
    Eigen::MatrixXd _trans;
    
    stacked_R.push_back(U*r_z_plus*transV);
    _trans = U*r_z_plus*sigma*transU;
    _trans.diagonal() << 0, 0, 0;
    stacked_ssmT.push_back(_trans);
    
    //1. R = U*Rz(+pi/2)*transV, ssm_T = U*Rz(-pi/2)sigma*U.transpose()
    stacked_R.push_back(U*r_z_plus*transV);
    _trans = U*r_z_minus*sigma*transU;
    _trans.diagonal() << 0, 0, 0;
    stacked_ssmT.push_back(_trans);

    //2. R = U*Rz(-pi/2)*transV, ssm_T = U*Rz(+pi/2)sigma*U.transpose()
    stacked_R.push_back(U*r_z_minus*transV);
    _trans = U*r_z_plus*sigma*transU;
    _trans.diagonal() << 0, 0, 0;
    stacked_ssmT.push_back(_trans);

    //3. R = U*Rz(-pi/2)*transV, ssm_T = U*Rz(-pi/2)sigma*U.transpose()
    stacked_R.push_back(U*r_z_minus*transV);
    _trans = U*r_z_minus*sigma*transU;
    _trans.diagonal() << 0, 0, 0;
    stacked_ssmT.push_back(_trans);

    if(_debug){
        Mats::iterator id_r = stacked_R.begin();
        Mats::iterator id_ssmT = stacked_ssmT.begin();

        unsigned int counter = 0;
        while(id_r!=stacked_R.end() || id_ssmT!=stacked_ssmT.end()){
            std::cout<<"R_"<<counter<<std::endl;
            std::cout<<*id_r<<std::endl;
            std::cout<<"T_"<<counter<<std::endl;
            std::cout<<*id_ssmT<<std::endl;
            id_ssmT++;
            id_r++;
            counter++;
        }
    }

    return {stacked_R, stacked_ssmT};
}

Eigen::Vector3d Skew2Vector(Eigen::MatrixXd & _trans){
    Eigen::Vector3d t;
    t<<(-1)*_trans(1, 2), _trans(0,2), (-1)*_trans(0,1);
    return t;
}
// Eigen::MatrixXd convert_Point2To

//ex4
//check the positive depth calculating epipolar constraint
Eigen::MatrixXd check_epipolar(std::pair<Mats, Mats> & _mats, Camera & cam1, Camera & cam2, Eigen::MatrixXd & pose_cam1, std::vector<bool> & result_satisfactly_RT,bool _debug){

    //obtain the pixel coordinates of the observed points
    Eigen::MatrixXd _ans_SE3;
    // std::vector<bool> result = {false, false, false, false};
    std::vector<Point2D> * _cam1_noramlize_coords = &(cam1.normalized_coords);
    std::vector<Point2D> * _cam2_noramlize_coords = &(cam2.normalized_coords);

    Eigen::MatrixXd _correct_SE3;

    Mats stacked_R = (_mats.first); 
    Mats stacked_T = (_mats.second);
    std::vector<bool> _flags = flag_corresp(cam1, cam2, DEBUG);

    Eigen::Vector4d _transformed_coords;
    Eigen::Vector4d _homo_coormat_KRTds;
    bool break_now = false;
    Eigen::Vector3d _cam1_Npoint;
    Eigen::MatrixXd _cam1_Npoint_skewMat;
    Eigen::Vector3d _cam2_Npoint;
    Eigen::MatrixXd _cam2_Npoint_skewMat;

    unsigned int _8points = 0;
    
    for(unsigned int i = 0; i<stacked_R.size(); i++){
        break_now = false;
        
        Eigen::MatrixXd _cam2_SE3 = Eigen::MatrixXd::Zero(3, 4);
        _cam2_SE3.block(0,0,3,3) = (stacked_R[i]);
        _cam2_SE3.block(0,3,3,1)<<Skew2Vector(stacked_T[i]);
        
        Eigen::MatrixXd _cam1_SE3 = pose_cam1;
        
        Eigen::MatrixXd _M1 = cam1.getKRT(_cam1_SE3);
        Eigen::MatrixXd _M2 = cam2.getKRT(_cam2_SE3);

        _8points = 0;

        std::cout<<"------------------Option_"<<i<<std::endl;

        std::cout<<"Cam"<<cam1.id<<std::endl;
        std::cout<<"KRT"<<std::endl;
        std::cout<<_M1<<std::endl;

        std::cout<<"Cam"<<cam2.id<<std::endl;
        std::cout<<"KRT"<<std::endl;
        std::cout<<_M2<<std::endl;

        std::cout<<"-----------------Depth of the points------------"<<std::endl;

        unsigned int _counter_depth_positive = 0;
        for(unsigned int s = 0; s<_flags.size(); s++){
            if(_8points >=8){
                break;
            }

            if(_flags[s]){
                Eigen::MatrixXd system_Mat = Eigen::MatrixXd::Zero(6, 4);
                //Trianglation
                Eigen::MatrixXd skewMat_cam1Point = cam1.normalized_coords[s].skew_symmetric_mat();
                Eigen::MatrixXd skewMat_cam2Point = cam2.normalized_coords[s].skew_symmetric_mat();
                system_Mat.block(0, 0, 3, 4) = skewMat_cam1Point * _M1;
                system_Mat.block(3, 0, 3, 4) = skewMat_cam2Point * _M2;
                std::cout<<"-----------"<<_8points<<"th point"<<"-----------"<<std::endl;
                // std::cout<<system_Mat<<std::endl;

                Eigen::JacobiSVD<Eigen::MatrixXd> svd(system_Mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

                svd.compute(system_Mat);

                Eigen::MatrixXd svd_transV = svd.matrixV();
                Eigen::VectorXd _point_in_cam1 = svd_transV.col(svd_transV.cols()-1);

                std::cout<<"Corresponding point in cam"<<cam1.id<<"coordinate"<<std::endl;
                std::cout<<_point_in_cam1<<std::endl;
                std::cout<<"Depth of the coordinate in cam"<<cam1.id<<std::endl;
                std::cout<<_point_in_cam1.z()<<std::endl;

                Eigen::VectorXd _point_in_cam2 = _cam2_SE3 * _point_in_cam1;

                std::cout<<"Corresponding point in cam"<<cam2.id<<"coordinate"<<std::endl;
                std::cout<<_point_in_cam2<<std::endl;
                std::cout<<"Depth of the coordinate in cam"<<cam2.id<<std::endl;
                std::cout<<_point_in_cam2.z()<<std::endl;

                if(_point_in_cam1.z() >= 0.0 && _point_in_cam2.z() >= 0.0){ //volume, or _transformed_coords.z()
                    _counter_depth_positive++;
                }
                _8points++;
            }
        }

        std::cout<<"System Matrix for triangulation"<<std::endl;

        if(_counter_depth_positive == 8){
            result_satisfactly_RT[i] = true;
            _correct_SE3 = _cam2_SE3;
        }

        if(result_satisfactly_RT[i]){
            std::cout<<"------------------Option_"<<i<<"is the correct R|T"<<std::endl;
        }else{
            std::cout<<"------------------Option_"<<i<<"is the wrong R|T"<<std::endl;
        }

    }


    if(_debug){

        for(unsigned int i = 0; i < 4; i++){
            if(!result_satisfactly_RT[i]){
                std::cout<<"Option"<<i<<"does not satisfy the requirement"<<std::endl;
            }else{
                std::cout<<"Option"<<i<<"is CORRECT SE3"<<std::endl;
                std::cout<<"Correct SE3"<<std::endl;
                std::cout<<_correct_SE3<<std::endl;

                // std::cout<<"Rotation"<<std::endl;
                // std::cout<<'\t'<<stacked_R[i]<<std::endl;        

                // std::cout<<"Translation"<<std::endl;
                // std::cout<<'\t'<<stacked_T[i]<<std::endl;

            }
        }

    }

    return _correct_SE3;
}

Eigen::MatrixXd find_relativePose(Camera & cam1, Camera & cam2, Eigen::MatrixXd & pose_cam1,bool _debug){


    //AlPHA
    Eigen::MatrixXd ALPHA_cam1_cam2 = kronocker_product(cam1, cam2, DEBUG);
    // std::cout<<ALPHA_cam1_cam2<<std::endl;

    //Derived ker(ALPHA) = E_s

    Eigen::MatrixXd System_mat = ALPHA_cam1_cam2.transpose() * ALPHA_cam1_cam2;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd1(System_mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

    svd1.compute(System_mat);

    Eigen::MatrixXd svd1_transV = svd1.matrixV();
    Eigen::VectorXd _original_Es = svd1_transV.col(svd1_transV.cols()-1);

    std::cout<<"Original Solution Es"<<std::endl;
    std::cout<<_original_Es<<std::endl;

    Eigen::VectorXd _check_original_Es = ALPHA_cam1_cam2 * _original_Es;
    std::cout<<"Check if the original Es is Kernel"<<std::endl;
    std::cout<<_check_original_Es<<std::endl;

    Eigen::MatrixXd _approx_Essential;
    _approx_Essential = map_Vect2Squaremat(_original_Es);

    std::cout<<"Obtain approx. Essential matrix"<<std::endl;
    std::cout<<_approx_Essential<<std::endl;

    //Map approx. Essential onto Essential space
    Eigen::JacobiSVD<Eigen::MatrixXd> svd2(_approx_Essential, Eigen::ComputeThinU | Eigen::ComputeThinV);

    //Modify the singular values to 1, 1, 0
    Eigen::MatrixXd _sv_EssesMap = Eigen::MatrixXd::Identity(3, 3);
    // Eigen::MatrixXd _sv_EssesMap = Eigen::MatrixXd::Zero(3, 3);
    // Eigen::Vector3d _singularValues = svd2.singularValues();

    // double average_variance = _singularValues.block(0,0,2,1).mean();
    // std::cout<<"Average variance: "<<average_variance<<std::endl;
    // _sv_EssesMap.diagonal() << average_variance, average_variance, 0.0;
    _sv_EssesMap(2, 2) = 0;
    // _sv_EssesMap.diagonal() = svd2.singularValues();
    std::cout<<"_sv_EssesMap"<<std::endl;
    std::cout<<_sv_EssesMap<<std::endl;

    Eigen::MatrixXd svd2_transV = svd2.matrixV();

    //Mapped essential matrix
    Eigen::MatrixXd _mapped_Es =svd2.matrixU() * _sv_EssesMap * svd2_transV;

    std::cout<<"Essential Matrix in essential space"<<std::endl;
    std::cout<<_mapped_Es<<std::endl;

    Mats potential_R;
    Mats potential_ssmT;

    std::pair<Mats, Mats> potential_RssmT;

    //obtain possible R|T solutions (4 combinations)
    //from camera 1 to camera 2
    potential_RssmT=possible_EuclidianTrans(svd2, _sv_EssesMap, DEBUG);

    // potential_R = potential_RssmT.first;
    // potential_ssmT = potential_RssmT.second;

    //check which one is the solution which has the positive depth 
    //when we apply the R|T to the normalize 2D point coordinate in the first camera 
    //from camera 1 to camera 2
    //{1, 0, 0, 1}
    std::vector<bool> _result_satisfactly_RT= {false, false, false, false};
    Eigen::MatrixXd _correct_SE3 = check_epipolar(potential_RssmT, cam1, cam2, pose_cam1, _result_satisfactly_RT, DEBUG);

    Eigen::MatrixXd _4x4_correct_SE3 = Eigen::MatrixXd::Identity(4, 4);
    _4x4_correct_SE3.block(0,0,3,4) = _correct_SE3;

    return _4x4_correct_SE3;
}

int main(int argc, char const *argv[])
{

    std::cout<<std::setprecision(6);
    unsigned int num_cam = 3;
    std::vector<Camera> cameras;
    SetCamParams(cameras, num_cam, DEBUG);


    std::vector<Point2D> pixel_coords;
    pixcoords_input_reader(pixel_coords, false);
    assign_pixelcoords2cameras(cameras, pixel_coords, DEBUG);


    // Eigen::MatrixXd dummy_mat(3, 3);
    // dummy_mat<<1, 2, 3,
    //         4, 5, 6,
    //         7, 8, 9;
    // //Essential matrix
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat_Essential;
    // mat_Essential = dummy_mat;
    // //Rowmajor-flatten 
    // //Map elements from essential matrix to the vector
    // //Stack elements in row-major
    // Eigen::Map<Eigen::RowVectorXd> E_s(mat_Essential.data(), mat_Essential.size());

    // std::cout<<"Essential Matrix E"<<std::endl;
    // std::cout<<mat_Essential<<std::endl;
    // std::cout<<"Stacked essential matrix elements in row-major fashion"<<std::endl;
    // std::cout<<E_s.transpose()<<std::endl;
    Eigen::MatrixXd cam1_SE3 = Eigen::MatrixXd::Zero(3, 4);
    cam1_SE3(0,0) = 1.0;
    cam1_SE3(1,1) = 1.0;
    cam1_SE3(2,2) = 1.0;
    Eigen::MatrixXd relativePose4x4_cam2_1 = find_relativePose(cameras[0], cameras[1], cam1_SE3, DEBUG);
    
    // Eigen::MatrixXd cam2_SE3 = relativePose4x4_cam2_1.block(0, 0, 3, 4);
    Eigen::MatrixXd cam2_SE3 = Eigen::MatrixXd::Zero(3,4);
    cam2_SE3(0,0) = 1.0;
    cam2_SE3(1,1) = 1.0;
    cam2_SE3(2,2) = 1.0;
    
    Eigen::MatrixXd relativePose4x4_cam1_0 = find_relativePose(cameras[1], cameras[2], cam2_SE3, DEBUG);

    std::cout<<"---------------------------------------"<<std::endl;
    std::cout<<"Relative pose from camera 0 to camera 1"<<std::endl;
    std::cout<<relativePose4x4_cam2_1<<std::endl;

    std::cout<<"Relative pose from camera 1 to camera 2"<<std::endl;
    std::cout<<relativePose4x4_cam1_0<<std::endl;

    std::cout<<"---------------------------------------"<<std::endl;

    Eigen::MatrixXd rotation_cam2_cam0 = relativePose4x4_cam2_1.block(0, 0, 3, 3) * relativePose4x4_cam1_0.block(0,0,3,3);
    std::cout<<rotation_cam2_cam0<<std::endl;

    Eigen::MatrixXd translation_cam2_cam0 = (-1) * relativePose4x4_cam2_1.block(0, 3, 3, 1) + (-1) * relativePose4x4_cam1_0.block(0, 3, 3, 1);
    std::cout<<translation_cam2_cam0<<std::endl;
    // //AlPHA
    // Eigen::MatrixXd ALPHA_cam1_cam2 = kronocker_product(cameras[0], cameras[1], DEBUG);
    // // std::cout<<ALPHA_cam1_cam2<<std::endl;

    // //Derived ker(ALPHA) = E_s

    // Eigen::MatrixXd System_mat = ALPHA_cam1_cam2.transpose() * ALPHA_cam1_cam2;
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd1(System_mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // svd1.compute(System_mat);

    // Eigen::MatrixXd svd1_transV = svd1.matrixV();
    // Eigen::VectorXd _original_Es = svd1_transV.col(svd1_transV.cols()-1);

    // std::cout<<"Original Solution Es"<<std::endl;
    // std::cout<<_original_Es<<std::endl;

    // Eigen::VectorXd _check_original_Es = ALPHA_cam1_cam2 * _original_Es;
    // std::cout<<"Check if the original Es is Kernel"<<std::endl;
    // std::cout<<_check_original_Es<<std::endl;

    // Eigen::MatrixXd _approx_Essential;
    // _approx_Essential = map_Vect2Squaremat(_original_Es);

    // std::cout<<"Obtain approx. Essential matrix"<<std::endl;
    // std::cout<<_approx_Essential<<std::endl;

    // //Map approx. Essential onto Essential space
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd2(_approx_Essential, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // //Modify the singular values to 1, 1, 0
    // Eigen::MatrixXd _sv_EssesMap = Eigen::MatrixXd::Identity(3, 3);
    // // Eigen::MatrixXd _sv_EssesMap = Eigen::MatrixXd::Zero(3, 3);
    // // Eigen::Vector3d _singularValues = svd2.singularValues();

    // // double average_variance = _singularValues.block(0,0,2,1).mean();
    // // std::cout<<"Average variance: "<<average_variance<<std::endl;
    // // _sv_EssesMap.diagonal() << average_variance, average_variance, 0.0;
    // _sv_EssesMap(2, 2) = 0;
    // // _sv_EssesMap.diagonal() = svd2.singularValues();
    // std::cout<<"_sv_EssesMap"<<std::endl;
    // std::cout<<_sv_EssesMap<<std::endl;

    // Eigen::MatrixXd svd2_transV = svd2.matrixV();

    // //Mapped essential matrix
    // Eigen::MatrixXd _mapped_Es =svd2.matrixU() * _sv_EssesMap * svd2_transV;

    // std::cout<<"Essential Matrix in essential space"<<std::endl;
    // std::cout<<_mapped_Es<<std::endl;

    // Mats potential_R;
    // Mats potential_ssmT;

    // std::pair<Mats, Mats> potential_RssmT;

    // //obtain possible R|T solutions (4 combinations)
    // //from camera 1 to camera 2
    // potential_RssmT=possible_EuclidianTrans(svd2, _sv_EssesMap, DEBUG);

    // // potential_R = potential_RssmT.first;
    // // potential_ssmT = potential_RssmT.second;

    // //check which one is the solution which has the positive depth 
    // //when we apply the R|T to the normalize 2D point coordinate in the first camera 
    // //from camera 1 to camera 2
    // //{1, 0, 0, 1}
    // std::vector<bool> _result_satisfactly_RT= {false, false, false, false};
    // Eigen::MatrixXd _SE3_cam02cam1 = check_epipolar(potential_RssmT, cameras[0], cameras[1], _result_satisfactly_RT, DEBUG);

    // std::cout<<"Correct SE3 from cam0 to cam1"<<std::endl;
    // std::cout<<_SE3_cam02cam1<<std::endl;


    return 0;
}

