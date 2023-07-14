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
    long double value;
};

class Matrix{

public:
    Eigen::Index dim;
    Eigen::MatrixXd mat;
    Eigen::MatrixXd identity;
    bool trans = false;

    Matrix(Eigen::Index d): dim(d){
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

    void mul(Eigen::Index & index, long double & scalar){
        base_mul(this->mat, index, scalar, true);
        base_mul(this->identity, index, scalar, false);
    }


    void add(Eigen::Index & i, Eigen::Index & j, long double & scalar){
        base_add(this->mat, i, j, scalar, true);
        base_add(this->identity, i, j, scalar, false);
    }

    void insert(Eigen::Index i, Eigen::Index j, long double value){
        mat(i,j) = value;
    }

    int __dimension__(){
        return dim;
    }
    
    static void base_swap(Eigen::MatrixXd & matrix, Eigen::Index & index_i, Eigen::Index & index_j, bool print){
        Eigen::VectorXd temp = matrix.row(index_j);
        matrix.row(index_j) =  matrix.row(index_i);
        matrix.row(index_i) =  temp;
        if (print && index_i != index_j){
            std::cout<<SWAP<<" "<<index_i<<" "<<index_j<<std::endl;
        }
    }

    static void base_mul(Eigen::MatrixXd & matrix, Eigen::Index & index, long double & scalar, bool print){
        matrix.row(index) *= scalar;
        if(print){
            std::cout<<MUL<<" "<<index<<" "<<scalar<<std::endl;
        }
    }

    static void base_add(Eigen::MatrixXd & matrix, Eigen::Index & index_i, Eigen::Index & index_j, long double & scalar, bool print){
        matrix.row(index_i) = matrix.row(index_i) + matrix.row(index_j) * scalar;
        Eigen::Index end_row = matrix.rows()-1;
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
    Eigen::Index _matrix_dimension;
    std::cin>>_matrix_dimension;
    Matrix matrix = {_matrix_dimension};
    Eigen::Index dim = matrix.__dimension__();
    long double copy = 0.0;
    for(Eigen::Index  i=0; i<dim; i++){
        for(Eigen::Index  j=0; j<dim; j++){
            std::cin>>copy;
            matrix.insert(i, j, copy);
        }
    }
    return matrix;
}

std::string DEGENERATE_checker(Matrix * matrix){
    mat_entry row_max;
    std::string ans = SOLUTION;
    Eigen::Index dim = matrix->dim;
    mat_entry rest_nonzero;

    for(Eigen::Index i = 0; i<dim; i++){
        row_max.row = i;
        row_max.value = matrix->mat.row(i).maxCoeff(&row_max.col);
        if(row_max.value == 0){
            // if(row_max.row < dim-1){
            //     Eigen::Index end_row = dim-1;
            //     for(Eigen::Index j = i+1; j<dim; j++){
            //         rest_nonzero.row = j;
            //         rest_nonzero.value = matrix->mat.row(j).maxCoeff(&rest_nonzero.col);
            //         if(rest_nonzero.value != 0){
            //             matrix->swap(i, j);
            //         }
            //     }
            // }
            ans = DEGENERATE;
            break;
        }
    }

    return ans;
}

std::string Gauss_elimination(Matrix * matrix){

    // Eigen::Index dim = matrix->dim;

    // long double one = 0.0;
    mat_entry max;
    mat_entry min;
    mat_entry row_max;
    mat_entry abs_max;
    Eigen::Index row = 0;
    Eigen::Index col = 0;
    std::string ans;

    ans = DEGENERATE_checker(matrix);
    if(ans == DEGENERATE){
        return ans;
    }

    while(row<matrix->dim && col<matrix->dim){
            max.col = col;
            min.col = col;
            Eigen::VectorXd target_col = matrix->mat.block(row, col, matrix->dim-row, 1);
            max.value = target_col.maxCoeff(&max.row);
            max.row +=row;
            min.value = target_col.minCoeff(&min.row);
            min.row +=row;

            if(std::abs(max.value) >= std::abs(min.value)){
                abs_max = max;
            }else{
                abs_max = min;
            }

            if(abs_max.value == matrix->mat(row, col) && abs_max.row != row){
                abs_max.row = row;
            }

            row_max.row = row;
            row_max.value = matrix->mat.row(row_max.row).maxCoeff(&row_max.col);

            if(abs_max.value == 0 && row_max.value == 0){
                row++;
                col++;
                continue;
            }

            if(abs_max.value == 0){
                // row++;
                col++;
                continue;
            }else{
                if(matrix->mat(row, col) == 0){
                    matrix->swap(row_max.row, abs_max.row);
                }
                
                if(matrix->mat(row, col) !=1){
                    long double factor = 1/matrix->mat(row, col);
                    matrix->mul(row, factor);
                }

                // matrix->swap(row, abs_max.row);
                for(Eigen::Index i = 0; i<matrix->dim; i++){
                    if(matrix->mat(i, col) == 0 || i == row){
                        continue;
                    }else{
                        long double scalar = (-1) * matrix->mat(i, col) / matrix->mat(row, col);
                        // matrix->mat(i, col) = 0;
                        matrix->add(i, row, scalar);
                    }
                }
                row++;
                col++;
            }
    }

    ans = DEGENERATE_checker(matrix);

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
