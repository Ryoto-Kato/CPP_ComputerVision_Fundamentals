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
    unsigned long int id;
    unsigned long int num_landmarks;
    unsigned long int dim;
    bool has_origin;
    std::vector<Landmark> lm_list;
    Eigen::MatrixXd lm_mat;
    bool null_space = false;

    Personal_space(Eigen::Index identifier){
        id = identifier;
    }

    ~ Personal_space(){
        lm_list.clear();
    }
    
    void init(unsigned long int d, unsigned long int n){
        num_landmarks = n;
        dim = d;
        lm_mat = Eigen::MatrixXd::Zero(num_landmarks, dim);

        for(unsigned long int num = 0; num < n; num++){
            Landmark lm;
            lm.name = n;
            lm.coordinate = Eigen::VectorXd::Zero(dim);
            lm_list.push_back(lm);
        }
        has_origin = false;

    }

    void vector2matrix(){
        for(unsigned long int i = 0; i<lm_list.size(); i++){
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

        // std::cout<<id<<" has origin: "<<has_origin<<std::endl;
        
    }

    void print_personal_space(){
        std::cout<<"Person"<<" "<<id<<std::endl;
        std::cout<<"Number of landmarks: "<<num_landmarks<<std::endl;
        std::cout<<"Dimension of space: "<<dim<<std::endl;
        for(unsigned long int i=0; i<num_landmarks; i++){
            std::cout<<lm_list[i].name<<" ";
            for(unsigned long int j = 0; j < dim; j++){
                std::cout<<lm_list[i].coordinate(j)<<" ";
            }
        std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
};

bool input_manager(Personal_space & p1, Personal_space & p2){

        Personal_space * p;
        unsigned long int d = 0;
        std::vector<Landmark> lm_name_list;
        double s = 1;
        bool null_space = false;

        for(unsigned int person = 0; person < 2; person++){
            if(person == 0){
                p = &p1;
                std::cin>>d;
                if(d < 1){
                    p1.null_space = true;
                    p2.null_space = true;
                    null_space = true;
                    break;
                }
            }else{
                p = &p2;
                s = -1;
            }

            unsigned long int nl;
            std::cin>>nl;
            if(nl<1){
                if(person == 0){
                    p1.null_space = true;
                    null_space = true;
                    break;
                }else{
                    p2.null_space = true;
                    null_space = true;
                    break;
                }
            }else{
                p->init(d, nl);

                double copy=0.0;

                std::vector<double> copy_list[d];
                std::string name_copy;
                Landmark temp1;

                for(unsigned long int i=0; i<p->num_landmarks; i++){
                    std::cin>>name_copy;
                    p->lm_list[i].name = name_copy;
                    for(unsigned long int j=0; j<p->dim; j++){
                        std::cin>>copy;
                        p->lm_list[i].coordinate(j) = copy * s;

                    }
                }

                p->vector2matrix();
            }
        }

        return null_space;

}

Eigen::VectorXd find_meeting_space(Personal_space & p1, Personal_space & p2){
    Eigen::MatrixXd lhs = Eigen::MatrixXd::Zero(p1.dim+2, p1.num_landmarks + p2.num_landmarks);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(p1.dim+2);
    rhs(p1.dim) = 1;
    rhs(p1.dim + 1) = 1;

    for(unsigned long int i = 0; i<p1.num_landmarks; i++){
        lhs.block(0, i, p1.dim, 1) = p1.lm_mat.transpose().col(i);
    }

    for(unsigned long int j = p1.num_landmarks; j < p1.num_landmarks + p2.num_landmarks; j++){
        lhs.block(0, j, p2.dim, 1) = p2.lm_mat.transpose().col(j-p1.num_landmarks);
    }


    lhs.block(p1.dim, 0, 1, p1.num_landmarks) = Eigen::VectorXd::Ones(p1.num_landmarks).transpose();


    lhs.block(p1.dim + 1, p1.num_landmarks, 1, p2.num_landmarks)  = Eigen::VectorXd::Ones(p2.num_landmarks).transpose();

    Eigen::MatrixXd System = lhs;
    // Eigen::MatrixXd RHS = (lhs.transpose() * rhs);
    Eigen::MatrixXd RHS = rhs;

    // std::cout<<System<<std::endl;
    // std::cout<<RHS<<std::endl;

    Eigen::FullPivLU<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> LU(System);
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd(System, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd m_coefficients;
	m_coefficients = LU.solve(RHS);
	// m_coefficients = svd.solve(RHS);

    Eigen::MatrixXd p1_base_matrix;
    Eigen::VectorXd p1_part_coefficients;
    Eigen::MatrixXd p2_base_matrix;
    Eigen::VectorXd p2_part_coefficients;

    p1_base_matrix = p1.lm_mat.transpose();
    p1_part_coefficients = m_coefficients.head(p1.num_landmarks);
    p2_base_matrix = p2.lm_mat.transpose();
    p2_part_coefficients = m_coefficients.tail(p2.num_landmarks);

    // std::cout<<"coefficients"<<std::endl;
    // std::cout<<m_coefficients.transpose()<<std::endl;

    Eigen::VectorXd p1_meeting_point = p1_base_matrix * p1_part_coefficients;
    Eigen::VectorXd p2_meeting_point = (-1)*p2_base_matrix * p2_part_coefficients;

    // std::cout<<"Meeting point p1"<<std::endl;
    // std::cout<<p1_meeting_point<<std::endl;
    // std::cout<<"Meeting point p2"<<std::endl;
    // std::cout<<p2_meeting_point<<std::endl;

    if(p1_meeting_point.squaredNorm() > p2_meeting_point.squaredNorm()){
        return p2_meeting_point;
    }else{
        return p1_meeting_point;
    }
}


int main(){

    Personal_space p1={1};
    Personal_space p2={2};

    bool zero_dim = input_manager(p1, p2);

    if(zero_dim){
        std::cout<<'N'<<std::endl;
    }else{
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

        // std::cout<<"meeting point"<<std::endl;
        // std::cout<<meeting_coordinate.transpose()<<std::endl;

        if(p1.has_origin && p2.has_origin){
            std::cout<<'Y'<<" ";
            std::cout<<meeting_coordinate.transpose()<<std::endl;
        }else{
            if(*abs_max_coef == 0){
                std::cout<<'N'<<std::endl;
            }else{
                std::cout<<'Y'<<" ";
                std::cout<<meeting_coordinate.transpose()<<std::endl;
            }
        }
    }


    return 0;
}