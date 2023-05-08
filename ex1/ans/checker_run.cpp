#include <iostream>
#include "checker.h"

using namespace std;


int main(int argc, char* argv[])
{
    if(argc != 3)
    {
        cout << "Usage: ./checker_run <file1> <file2>" << endl;
        exit(-1);
    }

    try {
        string in1 = argv[1];
        string in2 = argv[2];

        if(checker(in1, in2)) {
            cout << "Files are equal" << endl;
            exit(0);
        }
        else
        {
            cout << "Files are not equal" << endl;
            exit(1);
        }

    } catch (const exception& ex) {
        cerr << ex.what() << endl;
        exit(-2);
    }

    return -3;
}
