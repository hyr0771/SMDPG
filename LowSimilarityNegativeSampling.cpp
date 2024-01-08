#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <queue>
#include <istream>
#include <streambuf>
#include <fstream>
#include <sstream>
#include  <iomanip>
#include <algorithm>
#include <set>

using namespace std;

vector<vector<double>> Readcsv(string filename){

    ifstream csv_data(filename, std::ios::in);
    string line;

    if (!csv_data.is_open())
    {
        cout << "Error: opening file fail" << endl;
        exit(1);
    }

    istringstream sin;
    vector<double> words;
    string word;

    vector<vector<double>>res;
    while (getline(csv_data, line))
    {
        sin.clear();
        sin.str(line);
        words.clear();
        while (getline(sin, word, ','))
        {
            words.push_back(stod(word));
        }
        res.push_back(words);
    }
    csv_data.close();
    cout << "Read csv Run Ok......" << '\n';
    return res;
}


void Writecsv(string filename, vector<vector<double>>& res){
    std::ofstream outFile;
    outFile.open(filename, std::ios::out | std::ios::trunc);
    int m = res.size(), n = res[0].size();
    for(int i = 0; i < m; ++i){
        outFile << setprecision(10) << res[i][0];
        for(int j = 1; j < n; ++j){
            outFile << ',' << setprecision(10) << res[i][j];
        }
        outFile << endl;
    }
    outFile.close();
    cout << "Write csv Run Ok......" << '\n';
}

void zhuijia_Writecsv_two(string filename, vector<vector<double>>& res){
    std::ofstream outFile;
    outFile.open(filename, std::ios::out | std::ios::app);
    int m = res.size(), n = res[0].size();
    for(int i = 0; i < m; ++i){
        outFile << setprecision(10) << res[i][0];
        for(int j = 1; j < n; ++j){
            outFile << ',' << setprecision(10) << res[i][j];
        }
        outFile << endl;
    }
    outFile.close();
}

void zhuijia_Writecsv(string filename, vector<double>& res){
    std::ofstream outFile;
    outFile.open(filename, std::ios::out | std::ios::app);
    int n = res.size();
    outFile << setprecision(10) << res[0];
    for(int j = 1; j < n; ++j){
        outFile << ',' << setprecision(10) << res[j];
    }
    outFile << endl;
    outFile.close();
}



int main(){

    LowSimilarityNegativeSampling();
    return 0;
}
