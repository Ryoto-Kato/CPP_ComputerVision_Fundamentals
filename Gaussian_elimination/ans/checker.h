#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <cassert>

int length_checker(std::string & answer, std::string & my_result){
    bool check = true;
    int len_f_answer, len_f_result;

    std:: ifstream f_answer (answer, std::ifstream::binary);
    if(!f_answer){
        std::cout<<"Could not read file: "<<answer<<std::endl;
    }else{
        f_answer.seekg(0, f_answer.end);
        len_f_answer = f_answer.tellg();
        f_answer.seekg(0, f_answer.beg);
        std::cout<<"Length of file content: "<<len_f_answer<<std::endl;
    }

    f_answer.close();

    std:: ifstream f_result(my_result, std::ifstream::binary);

    if(!f_result){
        std::cout<<"Could not read file: "<<my_result<<std::endl;
    }else{
        f_result.seekg(0, f_result.end);
        len_f_result = f_result.tellg();
        f_result.seekg(0, f_result.beg); // bring the pointer back to the initial index
        std::cout<<"Length of file content: "<<len_f_result<<std::endl;
    }

    f_result.close();

    if(len_f_answer != len_f_result){
        std::cout<<"Length of output is different"<<std::endl;
        check = false;
    }

    return len_f_answer;
}

bool checker(std::string answer, std::string my_result){
    bool check = true;

    int len_check_result = length_checker(answer, my_result);

    if(len_check_result < 0){
        return false;
    }

    std:: ifstream f_answer (answer, std::ifstream::binary);
    std:: ifstream f_result (my_result, std::ifstream::binary);

    f_answer.seekg(0, f_answer.beg);
    f_result.seekg(0, f_result.beg);

    std::string copy_ans;
    std::string copy_result;

    std::string error;

    
    while(f_answer.tellg() <= f_answer.end && f_result.tellg() <= f_result.end){
            f_answer>>copy_ans;
            f_result>>copy_result;
            if(copy_ans != copy_result){
                check = false;
            }
            if(!check){
                error.append(copy_result);
            }
    }

    std::cout<<error<<std::endl;

    f_answer.close();
    f_result.close();

    return check;
}