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

void LowSimilarityNegativeSampling(){

    vector<vector<double>>met_sim = Readcsv("MSU.csv");
    vector<vector<double>>dis_sim = Readcsv("DSU.csv");
    vector<vector<double>>met_dis_matrix = Readcsv("MD.csv");


    int nd = dis_sim.size(), nm = met_sim.size();
    int n_association = 0;
    for(int i = 0; i < met_dis_matrix.size(); ++i)
        for(int j = 0; j < met_dis_matrix[0].size(); ++j){
            n_association += (int)met_dis_matrix[i][j];
        }

    printf("nd: %d, nm: %d, n_association: %d\n", nd, nm, n_association);

    set<int>meta_lowsim;
    set<int>dis_lowsim;
    int NN = 3;

    for(int i = 0; i < nm; ++i){
        priority_queue<vector<double>, vector<vector<double>> > q;
        for(int j = 0; j < nm; ++j){
            if(j < NN){
                q.push({met_sim[i][j], j});
            }else{
                vector<double>cnt = q.top();
                if(cnt[0] - met_sim[i][j] > 1e-5){
                    q.pop();
                    q.push({met_sim[i][j], j});
                }
            }
        }
        while(!q.empty()){
            vector<double> cnt = q.top();
            q.pop();
            meta_lowsim.insert(cnt[1]);
        }
    }

    for(int i = 0; i < nd; ++i){
        priority_queue<vector<double>, vector<vector<double>> > q;
        for(int j = 0; j < nd; ++j){
            if(j < NN){
                q.push({dis_sim[i][j], j});
            }else{
                vector<double>cnt = q.top();
                if(cnt[0] - dis_sim[i][j] > 1e-5){
                    q.pop();
                    q.push({dis_sim[i][j], j});
                }
            }
        }
        while(!q.empty()){
            vector<double> cnt = q.top();
            q.pop();
            dis_lowsim.insert(cnt[1]);
        }
    }

    int meta_lowsim_size = meta_lowsim.size();
    int dis_lowsim_size = dis_lowsim.size();
    printf("meta_lowsim: %d, dis_lowsim: %d,  nm *nd = %d\n", meta_lowsim_size, dis_lowsim_size, meta_lowsim_size * dis_lowsim_size);

    vector<vector<double>>pos_sample;
    vector<vector<double>>neg_sample;

    for(int i = 0; i < nm; ++i){
        for(int j = 0; j < nd; ++j){
            if((int)met_dis_matrix[i][j] == 1){
                pos_sample.push_back({(double)i, (double)j, 1});
            }
        }
    }

    for(auto x: meta_lowsim){
        for(auto y: dis_lowsim){
            if(met_dis_matrix[x][y] == 0){
                neg_sample.push_back({(double)x, (double)y, 0});
            }
        }
    }

    int n_pos = pos_sample.size(), n_neg = neg_sample.size();
    printf("pos_sample number: %d, neg_sample number: %d\n", n_pos, n_neg);

    for(int i = 0; i < 3; ++i){
        vector<int>random_index(n_neg);
        for(int i = 0; i < random_index.size(); ++i) random_index[i] = i;
        srand(i + 99);
        random_shuffle(random_index.begin(), random_index.end());   //打乱顺序

        cout << "random_index size: " << random_index.size() << '\n';

        //随机n_pos个作为负样本 + 间隔取样
        vector<vector<double>>ans;
        int deta = n_neg / n_pos;
        for(int j = 0; j < n_pos; ++j){
            ans.push_back(neg_sample[random_index[j * deta]]);
        }

        string neg_name = "NN3_1_1_balance_LWNegSample_" + to_string(i) + ".csv";
        Writecsv(neg_name, ans);
    }

}

int main(){

    LowSimilarityNegativeSampling();
    return 0;
}
