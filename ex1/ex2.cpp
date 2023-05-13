#include <iostream>
#include <Eigen/Dense>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>

struct Landmark{
    std::string name;
    Eigen::VectorXd coordinate;
};

class Personal_space{
public:
    int id;
    int num_landmarks;
    int dim;
    bool has_origin;
    std::vector<Landmark> lm_list;
    Eigen::MatrixXd lm_mat;
    std::vector<double> double_affine_space;

    Personal_space(int identifier){
        id = identifier;
    }

    ~ Personal_space(){
        lm_list.clear();
        double_affine_space.clear();
    }
    
    void init(int d, int n){
        num_landmarks = n;
        dim = d;
        lm_mat = Eigen::MatrixXd::Zero(num_landmarks, dim);

        for(int num = 0; num < n; num++){
            Landmark lm;
            lm.name = n;
            lm.coordinate = Eigen::VectorXd::Zero(dim);
            lm_list.push_back(lm);
        }

        for(int d = 0; d<dim; d++){
            double_affine_space.push_back(1.0);
        }
        has_origin = false;

    }

    void vector2matrix(){
        for(int i = 0; i<lm_list.size(); i++){
            lm_mat.row(i) = lm_list[i].coordinate;
            double min = lm_mat.row(i).minCoeff();
            double max = lm_mat.row(i).maxCoeff();
            double * abs_max;

            if(std::abs(max) >= std::abs(min)){
                abs_max = &max;
            }else{
                abs_max = &min;
            }
            if(*abs_max == 0){
                has_origin = true;
            }
        }
        
    }

    void print_personal_space(){
        std::cout<<"Person"<<" "<<id<<std::endl;
        std::cout<<"Number of landmarks: "<<num_landmarks<<std::endl;
        std::cout<<"Dimension of space: "<<dim<<std::endl;
        for(int i=0; i<num_landmarks; i++){
            std::cout<<lm_list[i].name<<" ";
            for(int j = 0; j < dim; j++){
                std::cout<<lm_list[i].coordinate(j)<<" ";
            }
        std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
};

void input_manager(Personal_space & p1, Personal_space & p2){

        Personal_space * p;
        int d = 0;
        std::vector<Landmark> lm_name_list;

        for(int person = 0; person < 2; person++){
            if(person == 0){
                p = &p1;
                std::cin>>d;
            }else{
                p = &p2;
            }

            int nl;
            std::cin>>nl;

            p->init(d, nl);

            double copy=0.0;

            std::vector<double> copy_list[d];
            std::string name_copy;
            Landmark temp1;

            for(int i=0; i<p->num_landmarks; i++){
                std::cin>>name_copy;
                p->lm_list[i].name = name_copy;
                for(int j=0; j<p->dim; j++){
                    std::cin>>copy;
                    p->lm_list[i].coordinate(j) = copy;

                }
            }

            p->vector2matrix();
        }
}

Eigen::VectorXd find_meeting_space(Personal_space & p1, Personal_space & p2){
    Eigen::MatrixXd lhs = Eigen::MatrixXd::Zero(p1.dim+2, p1.num_landmarks + p2.num_landmarks);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(p1.dim+2);
    rhs(p1.dim) = 1;
    rhs(p1.dim + 1) = 1;

    for(int i = 0; i<p1.num_landmarks; i++){
        lhs.block(0, i, p1.dim, 1) = p1.lm_mat.transpose().col(i);
    }

    for(int j = p1.num_landmarks; j < p1.num_landmarks + p2.num_landmarks; j++){
        lhs.block(0, j, p2.dim, 1) = p2.lm_mat.transpose().col(j-p1.num_landmarks);
    }


    lhs.block(p1.dim, 0, 1, p1.num_landmarks) = Eigen::VectorXd::Ones(p1.num_landmarks).transpose();


    lhs.block(p1.dim + 1, p1.num_landmarks, 1, p2.num_landmarks)  = Eigen::VectorXd::Ones(p2.num_landmarks).transpose();

    Eigen::MatrixXd System = lhs;
    // Eigen::MatrixXd RHS = (lhs.transpose() * rhs);
    Eigen::MatrixXd RHS = rhs;

    Eigen::FullPivLU<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> LU(System);
    Eigen::VectorXd m_coefficients;
	m_coefficients = LU.solve(RHS);

    Eigen::MatrixXd base_matrix;
    Eigen::VectorXd part_coefficients;

    if(p1.has_origin){
        base_matrix = p2.lm_mat.transpose();
        part_coefficients = m_coefficients.tail(p2.num_landmarks);
    }else if(p2.has_origin){
        base_matrix = p1.lm_mat.transpose();
        part_coefficients = m_coefficients.head(p1.num_landmarks);
    }else{
        base_matrix = p2.lm_mat.transpose();
        part_coefficients = m_coefficients.tail(p2.num_landmarks);
    }

    Eigen::VectorXd meeting_point = base_matrix * part_coefficients;

    return meeting_point;
}


int main(){

    Personal_space p1={1};
    Personal_space p2={2};

    input_manager(p1, p2);

    Eigen::VectorXd meeting_coordinate;

    meeting_coordinate = find_meeting_space(p1, p2);

    double max_coefficient = meeting_coordinate.maxCoeff();
    double min_coefficient = meeting_coordinate.minCoeff();
    double * abs_max_coef;

    if(std::abs(max_coefficient) >= std::abs(min_coefficient)){
        abs_max_coef = &max_coefficient;
    }else{
        abs_max_coef = &min_coefficient;
    }

    if(*abs_max_coef == 0){
        std::cout<<'N'<<std::endl;
    }else{
        std::cout<<'Y'<<" ";
        std::cout<<meeting_coordinate.transpose()<<std::endl;
    }

    return 0;
}