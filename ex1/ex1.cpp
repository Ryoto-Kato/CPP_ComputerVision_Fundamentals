#include <iostream>
#include <Eigen/Dense>
#include <sstream>
#include <string>
#include <iomanip>

#define SWAP 'S'
#define MUL 'M'
#define ADD 'A'
#define SOLUTION "SOLUTION"
#define DEGENERATE "DEGENERATE"
#define PRINT "PRINT"


struct mat_entry{
    Eigen::Index row;
    Eigen::Index col;
    double value;
};

class Matrix{

public:
    int dim;
    Eigen::MatrixXd mat;
    Eigen::MatrixXd identity;
    bool trans = false;

    Matrix(int d): dim(d){
        mat = Eigen::MatrixXd::Zero(d, d);
        identity = Eigen::MatrixXd::Identity(d, d);
    }

    Matrix(Eigen::MatrixXd & m, Eigen::MatrixXd & i, bool flag){
        mat = m;
        identity = i;
        trans = flag;
    }

    void swap(Eigen::Index & i, Eigen::Index & j){
        base_swap(this->mat, i, j, true);
        base_swap(this->identity, i, j, false);
    }

    void mul(Eigen::Index & index, double & scalar){
        base_mul(this->mat, index, scalar, true);
        base_mul(this->identity, index, scalar, false);
    }


    void add(Eigen::Index & i, Eigen::Index & j, double & scalar){
        base_add(this->mat, i, j, scalar, true);
        base_add(this->identity, i, j, scalar, false);
    }

    void insert(int i, int j, double value){
        mat(i,j) = value;
    }

    int __dimension__(){
        return dim;
    }
    
    static void base_swap(Eigen::MatrixXd & matrix, Eigen::Index & index_i, Eigen::Index & index_j, bool print){
        Eigen::VectorXd temp = matrix.row(index_j);
        matrix.row(index_j) =  matrix.row(index_i);
        matrix.row(index_i) =  temp;
        if (print){
            std::cout<<SWAP<<" "<<index_i<<" "<<index_j<<std::endl;
        }
    }

    static void base_mul(Eigen::MatrixXd & matrix, Eigen::Index & index, double & scalar, bool print){
        matrix.row(index) *= scalar;
        if(print){
            std::cout<<MUL<<" "<<index<<" "<<scalar<<std::endl;
        }
    }

    static void base_add(Eigen::MatrixXd & matrix, Eigen::Index & index_i, Eigen::Index & index_j, double & scalar, bool print){
        matrix.row(index_i) = matrix.row(index_i) + matrix.row(index_j) * scalar;
        if(print){
            std::cout<<ADD<<" "<<index_i<<" "<<index_j<<" "<<scalar<<std::endl;
        }
    }

    bool invertible(){
        return Eigen::MatrixXd::Identity(dim, dim) == identity;
    }

    void print(std::string s){
        /*print Solution or Degenerate (if the rank is not full rank)

        Args:
        ----
        s:    str, s from [SOLUTION, DEGENERATE]
        */

        if(s==PRINT){
            std::cout<<mat<<std::endl;
        }else{
            std::cout<<s<<std::endl;
            if(s == SOLUTION){
                std::cout<<identity<<std::endl;
                std::cout<<""<<std::endl;
            }
        }
    }
};

Matrix input_manager(){
    int _matrix_dimension;
    std::cin>>_matrix_dimension;
    Matrix matrix = {_matrix_dimension};
    int dim = matrix.__dimension__();
    double copy = 0.0;
    for(int i=0; i<dim; i++){
        for(int j=0; j<dim; j++){
            std::cin>>copy;
            matrix.insert(i, j, copy);
        }
    }
    return matrix;
}

std::string DEGENERATE_checker(Matrix * matrix, bool final_check){
    mat_entry row_max;
    std::string ans = SOLUTION;

    for(Eigen::Index i = 0; i<matrix->dim; i++){
        row_max.row = i;
        row_max.value = matrix->mat.row(i).maxCoeff(&row_max.col);
        if(row_max.value == 0){
            if(row_max.row != matrix->dim-1 && final_check){
                matrix->swap(row_max.row, ++i);
            }
            ans = DEGENERATE;
            break;
        }
    }

    return ans;
}

std::string Gauss_elimination(Matrix * matrix){

    double one = 1.0;
    double zero = 0.0;
    int dim = matrix->dim;
    double scaler = 0.0;
    mat_entry max;
    mat_entry min;
    mat_entry abs_max;
    Eigen::Index row = 0;
    Eigen::Index col = 0;
    Eigen::Index max_row_index = dim-1;
    std::string ans;

    while(row<matrix->dim && col<matrix->dim){

            max.col = col;
            min.col = col;
            Eigen::VectorXd target_row = matrix->mat.block(row, col, matrix->dim-row, 1);
            #ifdef DEBUG
            std::cout<<"target_row"<<std::endl;
            std::cout<<target_row<<std::endl;
            std::cout<<std::endl;
            #endif
            max.value = target_row.maxCoeff(&max.row);
            max.row +=row;
            min.value = target_row.minCoeff(&min.row);
            min.row +=row;


            if(std::abs(max.value) >= std::abs(min.value)){
                abs_max = max;
            }else{
                abs_max = min;
            }

            if(abs_max.value == matrix->mat(row, col) && abs_max.row != row){
                abs_max.row = row;
            }

            if (abs_max.value == 0){
                if(row == matrix->dim-1){
                    ans = DEGENERATE_checker(matrix, true);
                }else{
                    ans = DEGENERATE_checker(matrix, false);
                }
                if(ans == DEGENERATE){
                    break;
                }else{
                    // row++;
                    col++;
                    continue;
                }
            }else{
                if(matrix->mat(row, col) == 0){
                    matrix->swap(row, abs_max.row);
                    // continue;
                }
                if(matrix->mat(row, col) !=1){
                    double factor = 1/matrix->mat(row, col);
                    matrix->mul(row, factor);
                }

                for(Eigen::Index i = 0; i<matrix->dim; i++){
                    if(matrix->mat(i, col) == 0 || i == row){
                        continue;
                    }else{
                        double scalar = (-1) * matrix->mat(i, col) / matrix->mat(row, col);
                        matrix->add(i, row, scalar);
                    }
                }
            }
            // // #ifdef DEBUG
            // matrix->print(PRINT);
            // // #endif
            row++;
            col++;
    }

    ans = DEGENERATE_checker(matrix, true);

    return ans;
}

std::string inverse_matrix_gauss_elim(Matrix * A){
    std::string ans = "";
    ans = Gauss_elimination(A);

    return ans;
}

int main(int argc, char const *argv[])
{
    std::cout<<std::setprecision(20);

    Matrix A = input_manager();

    std::string answer = inverse_matrix_gauss_elim(&A);
    A.print(answer);

    return 0;
}
