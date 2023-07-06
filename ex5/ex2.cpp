#include <iostream>
#include <Eigen/Dense>
#include <sstream>
#include <string>
#include <iomanip>
#include <vector>
#include <memory>
#include <iterator>

#define DEBUG 1

auto print_vect=[](auto const & v){
    std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout<<std::endl;
};

Eigen::Vector3d se3_Skew2Vector(Eigen::MatrixXd & _transform){
    Eigen::VectorXd t = Eigen::VectorXd::Zero(6);
    //translation3dof and rotation 3dof
    t<<_transform(0, 3), _transform(1, 3), _transform(2, 3), _transform(2, 1), _transform(0, 2), _transform(1, 0);
    return t;
}

Eigen::MatrixXd se3_Vector2Skew(std::vector<double> & vect){
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(4, 4);
    double * t_x = &vect[0];
    double * t_y = &vect[1];
    double * t_z = &vect[2];
    double * alpha = &vect[3];
    double * beta = &vect[4];
    double * gamma = &vect[5];
    mat << 0, (-1)*(*gamma), (*beta), *t_x,
        (*gamma), 0, (-1)*(*alpha), *t_y,
        -1*(*beta), (*alpha), 0, *t_z,
        0, 0, 0, 1;
    return mat;
}

Eigen::Matrix3d Vector2SkewMat(Eigen::Vector3d & vector){
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(3, 3);
    mat << 0, (-1)*vector(2), vector(1),
        vector(2), 0, (-1)*vector(0),
        (-1)*vector(1), vector(0), 0;
    return mat;
}


Eigen::VectorXd standardVect2eigenVect(std::vector<double> & std_vect){
    unsigned int num_vect = std_vect.size();
    Eigen::VectorXd  eigenVect = Eigen::VectorXd::Zero(num_vect);
    for(unsigned int i = 0; i<num_vect; i++){
        eigenVect(i) = std_vect[i];
    }
    return eigenVect;
}

Eigen::MatrixXd se3_Vector2Skew(Eigen::VectorXd & vect){
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(4, 4);
    double * t_x = &vect[0];
    double * t_y = &vect[1];
    double * t_z = &vect[2];
    double * alpha = &vect[3];
    double * beta = &vect[4];
    double * gamma = &vect[5];
    mat << 0, (-1)*(*gamma), (*beta), *t_x,
        (*gamma), 0, (-1)*(*alpha), *t_y,
        -1*(*beta), (*alpha), 0, *t_z,
        0, 0, 0, 1;
    return mat;
}

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
        std::cout<<std::right<<x<<" "<<y<<" "<<std::endl;
    }

    Eigen::Vector3d eigen_vector(){
        Eigen::Vector3d _a = {x, y, depth};
        return _a;
    }

    Eigen::Vector2d vect_uv_coords(){
        Eigen::Vector2d _uv = {x, y};
        return _uv;
    }

    double norm(){
        // This should be applied after we devide the x and y by z
        double _norm = std::sqrt((x*x+y*y));
        return _norm;
    }
};

void pixcoords_input_reader(std::vector<Point2D> & points_f1, unsigned int num_points, bool _print){
    std::vector<double> line;
    double intermediate;

    while(std::cin>>intermediate){
        line.push_back(intermediate);
    }

    int counter = 0;
    std::vector<double> copy;
    for(unsigned int i = 0; i<2*num_points; i++){
        copy.push_back(line[i]);
        if((i+1)%2 == 0){
            Point2D _point_f1 = {copy};            
            points_f1.push_back(_point_f1);
            copy.erase(copy.begin(), copy.end());
            if(_print){
                _point_f1.print();
            }
        }
    }
}

void points2D_with_camera_extrinsic_and_intrinsic_reader(std::vector<Point2D> & points_f1, Eigen::MatrixXd & intrinsic, Eigen::MatrixXd & extrinsic, unsigned int num_points, bool _print){
    std::vector<double> line;
    double intermediate;

    while(std::cin>>intermediate){
        line.push_back(intermediate);
    }

    //read
    int counter = 0;
    std::vector<double> _copy_list;
    for(unsigned int i = 0; i<2*num_points; i++){
        _copy_list.push_back(line[i]);
        if((i+1)%2 == 0){
            Point2D _point_f1 = {_copy_list};            
            points_f1.push_back(_point_f1);
            _copy_list.erase(_copy_list.begin(), _copy_list.end());
            if(_print){
                _point_f1.print();
            }
        }
        counter++;
    }

    int offset = counter;

    for(unsigned int r = 0; r<3; r++){
        for(unsigned int c = 0; c<3; c++){
            intrinsic(r, c) = line[3*r+c + offset]; 
            counter++;
        }
    }

    offset = counter;

    for(unsigned int _r = 0; _r<3; _r++){
        for(unsigned int _c = 0; _c<4; _c++){
            extrinsic(_r, _c) = line[4*_r+_c + offset]; 
        }
    }

    if(_print){
        std::cout<<"input Intrinsic and Extrinsic"<<std::endl;
        std::cout<<intrinsic<<std::endl;
        std::cout<<extrinsic<<std::endl;
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
    Eigen::MatrixXd extrinsic;
    Image_plane_size _image_plane_size;
    Intrinsic_params _intrinsic_params;

    Camera() = default;

    Camera(Eigen::MatrixXd & _intrinsic, Eigen::MatrixXd & _extrinsic): intrinsic(_intrinsic), extrinsic(_extrinsic), _image_plane_size(0, 0), _intrinsic_params(0, 0, 0, 0){
        _intrinsic_params._focal.x = intrinsic(0, 0);
        _intrinsic_params._focal.y = intrinsic(1, 1);
        _intrinsic_params._center.x = intrinsic(0, 2);
        _intrinsic_params._center.y = intrinsic(1, 2);
    }

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

        std::cout<<"extrinsic matrix"<<std::endl;
        std::cout<<extrinsic<<std::endl;

        std::cout<<"----------------------------------------"<<std::endl;
    }
};

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

Eigen::MatrixXd se3toSE3(std::vector<double> & vect_se3,bool _debug){
    Eigen::MatrixXd sigma_hat = se3_Vector2Skew(vect_se3);
    Eigen::MatrixXd mat_SE3;
    if(sigma_hat.determinant() == 0){
        mat_SE3= Eigen::MatrixXd::Identity(3, 4);
    }else{
        Eigen::MatrixXd w_hat = sigma_hat.block(0, 0, 3, 3);
        Eigen::Vector3d translation_se3;
        translation_se3<<vect_se3[0], vect_se3[1], vect_se3[2];
        Eigen::Vector3d rotation_se3;
        rotation_se3<<vect_se3[3], vect_se3[4], vect_se3[5];  
        double norm_rotation_vect_se3 = rotation_se3.norm();
        double sin_norm_w = sin(norm_rotation_vect_se3);
        double cos_norm_w = cos(norm_rotation_vect_se3);

        Eigen::MatrixXd identity = Eigen::MatrixXd::Ones(3, 3);
        //exp_w_hat = 3x3
        Eigen::MatrixXd exp_w_hat;
        exp_w_hat = identity + (w_hat/norm_rotation_vect_se3) * sin_norm_w + (w_hat/norm_rotation_vect_se3)*(w_hat/norm_rotation_vect_se3) * (1 - cos_norm_w);

        //translation = 3x1
        Eigen::Vector3d exp_translation;
        exp_translation = ((identity - exp_w_hat) * w_hat * translation_se3 + rotation_se3 * rotation_se3.transpose() * translation_se3)/(norm_rotation_vect_se3 * norm_rotation_vect_se3);

        //final SE3
        mat_SE3 = Eigen::MatrixXd::Zero(3, 4);
        mat_SE3.block(0, 0, 3, 3) = exp_w_hat;
        mat_SE3.block(0, 3, 3, 1) = exp_translation;

        if(_debug){
            std::cout<<"skew symmetric matrix of se3"<<std::endl;
            std::cout<<sigma_hat<<std::endl;
            
            std::cout<<"SE3"<<std::endl;
            std::cout<<mat_SE3<<std::endl;
        }
    }
    return mat_SE3;
}

Eigen::MatrixXd getJacobian(std::vector<Point3D> world_coords, Eigen::MatrixXd mat_SE3, Camera & cam, bool _debug){
    
    unsigned int num_vertex = world_coords.size();
    if(_debug){
        std::cout<<"Number of vertex"<<std::endl;
        std::cout<<num_vertex<<std::endl;
    }
    Eigen::MatrixXd Jacobian = Eigen::MatrixXd::Zero(16, 10);

    for(unsigned int i = 0; i < num_vertex; i++){
        
        Eigen::MatrixXd dr_camParam = Eigen::MatrixXd::Zero(2, 4);
        
        //DP
        Eigen::VectorXd DP4 = cam.extrinsic * world_coords[i]._vector(); //4x1, homogeneous coordinate
        Eigen::Vector3d DP3 = {DP4.x(), DP4.y(), DP4.z()};

        if(_debug){
            std::cout<<"DP4"<<std::endl;
            std::cout<<DP4<<std::endl;
            std::cout<<"DP3"<<std::endl;
            std::cout<<DP3<<std::endl;
        }
        //EDP
        Eigen::VectorXd EDP4 = mat_SE3 * cam.extrinsic * world_coords[i]._vector();
        Eigen::Vector3d EDP3 = {EDP4.x(), EDP4.y(), EDP4.z()};


        if(_debug){
            std::cout<<"EDP4"<<std::endl;
            std::cout<<EDP4<<std::endl;
            std::cout<<"EDP3"<<std::endl;
            std::cout<<EDP3<<std::endl;
        }

        //set derivative in the first 4 columns
        dr_camParam(0, 0) = EDP3.x()/EDP3.z();
        dr_camParam(0, 1) = 0.0;
        dr_camParam(0, 2) = 1.0;
        dr_camParam(0, 3) = 0.0;

        dr_camParam(1, 0) = 0.0;
        dr_camParam(1, 1) = EDP3.y()/EDP3.z();
        dr_camParam(1, 2) = 0.0;
        dr_camParam(1, 3) = 1.0;

        if(_debug){
            std::cout<<"dr_camParam"<<std::endl;
            std::cout<<dr_camParam<<std::endl;
        }

        //dr_dedp
        //shape 2x3
        // Eigen::MatrixXd cam_intrinsic = cam.intrinsic;
        //translation part should be 0
        // cam_intrinsic(0, 2) = 0.0;
        // cam_intrinsic(1, 2) = 0.0;
        // cam_intrinsic(2, 2) = 0.0;

        double f_x = cam._intrinsic_params._focal.x;
        double f_y = cam._intrinsic_params._focal.y;

        Eigen::MatrixXd dr_dEDP = Eigen::MatrixXd::Zero(2, 3);
        dr_dEDP << f_x/EDP3.z(), 0.0, (-1)*(f_x*EDP3.x())/(EDP3.z()*EDP3.z()),
                0.0, f_y/EDP3.z(), (-1)*(f_y*EDP3.y())/(EDP3.z()*EDP3.z());

        // Eigen::MatrixXd dr_dEDP = cam_intrinsic;

        if(_debug){
            std::cout<<"dr_dEDP"<<std::endl;
            std::cout<<dr_dEDP<<std::endl;
        }

        //dedp_deps
        Eigen::Matrix3d skew_DP3 = Vector2SkewMat(DP3);
        skew_DP3 *= -1;
        Eigen::MatrixXd dEDP_deps = Eigen::MatrixXd::Zero(3, 6);

        if(_debug){
            std::cout<<"skew_DP3"<<std::endl;
            std::cout<<skew_DP3<<std::endl;
        }

        dEDP_deps(0,0) = 1.0;
        dEDP_deps(1,1) = 1.0;
        dEDP_deps(2,2) = 1.0;
        dEDP_deps.block(0, 3, 3, 3) = skew_DP3;

        if(_debug){
            std::cout<<"dEDP_deps"<<std::endl;
            std::cout<<dEDP_deps<<std::endl;
        }

        Eigen::MatrixXd dr_deps = dr_dEDP * dEDP_deps;
        if(_debug){
            std::cout<<"dr_deps"<<std::endl;
            std::cout<<dr_deps<<std::endl;
        }

        // set derivatives 
        Jacobian.block(2*i, 0, 2, 4) = dr_camParam;
        Jacobian.block(2*i, 4, 2, 6) = dr_deps;
    }

    if(_debug){
        std::cout<<"Jacobian"<<std::endl;
        std::cout<<Jacobian<<std::endl;
    }

    return Jacobian;
}

Eigen::MatrixXd instant_camIntrinsic_generator (Eigen::Vector4d & cam_intrinsic){
    //fx, fy, cx, cy
    Eigen::MatrixXd cam_intrinsic_mat = Eigen::MatrixXd::Identity(3, 3);
    cam_intrinsic_mat(0, 0) = cam_intrinsic[0];
    cam_intrinsic_mat(1, 1) = cam_intrinsic[1];
    cam_intrinsic_mat(0, 2) = cam_intrinsic[2];
    cam_intrinsic_mat(1, 2) = cam_intrinsic[3];

    return cam_intrinsic_mat;
}

int main(int argc, char const *argv[])
{

    std::cout<<std::setprecision(16);

    // std::cout<<"Input projected vertices coordinate on 2d image plane"<<std::endl;
    // std::cout<<"u"<<" "<<"v"<<std::endl;

    // obtain 3D cube vertices coordinate in world frame
    int num_vertex = 8;
    std::vector<Point3D> world_coords;
    world_coords = cube_config(num_vertex, DEBUG);

    // Obtain 2D projected cube vertices coordinate on 2d image plane
    std::vector<Point2D> pixel_coords;
    // pixcoords_input_reader(pixel_coords, num_vertex, DEBUG);

    // read intrinsic and extrinsic from the input
    Eigen::MatrixXd Intrinsic = Eigen::MatrixXd::Zero(3, 3);
    Eigen::MatrixXd Extrinsic = Eigen::MatrixXd::Identity(4, 4);
    points2D_with_camera_extrinsic_and_intrinsic_reader(pixel_coords, Intrinsic, Extrinsic, num_vertex, DEBUG);
    
    // Camera initialization
    Camera cam(Intrinsic, Extrinsic);

    // Check if the reader works
    if(DEBUG){
        cam.print_params();
    }

    // get vector containing intrinsic parameter (4 dim) 
    // fx, fy, cx, cy
    Eigen::Vector4d intrinsic_paramVect = {cam._intrinsic_params._focal.x, cam._intrinsic_params._focal.y, cam._intrinsic_params._center.x, cam._intrinsic_params._center.y};

    //t_x, t_y, t_z, alpha, beta, gamma;
    std::vector<double> vect_se3 = {0, 0, 0, 0, 0, 0};
    // //Exponential map of se3 to get SE3
    Eigen::MatrixXd mat_SE3 = se3toSE3(vect_se3, DEBUG);

    //where sigma = 0
    // Eigen::MatrixXd mat_SE3 = Eigen::MatrixXd::Identity(3, 4);
    
    if(DEBUG){
        std::cout<<"Mat SE3"<<std::endl;
        std::cout<<mat_SE3<<std::endl;
    }

    //pixel coordinate and world coordinate

    // residuals
    std::vector<double> residual_list;
    for(unsigned int i = 0; i < num_vertex; i++){
        // std::cout<<(cam.intrinsic * (matSE3 * cam.extrinsic * world_coords[i]._vector()))<<std::endl;
        // std::cout<<(cam.intrinsic * (mat_SE3 * cam.extrinsic * world_coords[i]._vector()))<<std::endl;
        Eigen::VectorXd projected_image_coords = cam.intrinsic * (mat_SE3 * cam.extrinsic * world_coords[i]._vector());
        Eigen::Vector2d projected_uv_coords;
        projected_uv_coords << projected_image_coords[0]/projected_image_coords[2], projected_image_coords[1]/projected_image_coords[2]; 
        Eigen::VectorXd _ith_residual = projected_uv_coords - pixel_coords[i].vect_uv_coords();
        // double norm_ith_residual = _ith_residual.norm();
        residual_list.push_back(_ith_residual.x());
        residual_list.push_back(_ith_residual.y());
    }


    Eigen::VectorXd residual_vect = standardVect2eigenVect(residual_list);

    Eigen::MatrixXd Jacobian = getJacobian(world_coords, mat_SE3, cam, DEBUG);

    //get delta vector (10 dim) (this is already multiplied with minus)
    Eigen::VectorXd delta = (-1) * (Jacobian.transpose() * Jacobian).inverse() * Jacobian.transpose() * residual_vect;

    if(DEBUG){
        std::cout<<"delta"<<std::endl;
        std::cout<<delta<<std::endl;
    }

    // update intrinsic parameter
    Eigen::Vector4d updated_intrinsic_paramVect = intrinsic_paramVect + delta.block(0, 0, 4, 1);
    Eigen::MatrixXd update_intrinsic_paramMat = instant_camIntrinsic_generator(updated_intrinsic_paramVect);
    
    if(DEBUG){
        std::cout<<"Updated intrinsic params"<<std::endl;
        std::cout<<update_intrinsic_paramMat<<std::endl;
    }

    if(DEBUG){
        std::cout<<"residual list"<<std::endl;
        std::cout<<residual_vect.transpose()<<std::endl;
    }

    // camera_calibration(world_coords, pixel_coords, DEBUG);
    return 0;
}

