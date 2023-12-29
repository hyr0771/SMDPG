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

void MDPGTConstruction(){

    vector<vector<double>>met_sim = Readcsv("MSM.csv");
    vector<vector<double>>dis_sim = Readcsv("DSM.csv");

    vector<vector<double>>met_dis_matrix = Readcsv("MD.csv");

    vector<vector<double>>pos_sample = Readcsv("pos_sample.csv");
    vector<vector<double>>neg_sample = Readcsv("NN3_1_1_balance_LWNegSample_0.csv");

    int nd = dis_sim.size(), nm = met_sim.size();
    int n_association = pos_sample.size();

    vector<vector<double>>node_sample = pos_sample;
    node_sample.insert(node_sample.end(), neg_sample.begin(), neg_sample.end());
    int sample_number = node_sample.size();

    vector<int>random_index(sample_number);
    for(int i = 0; i < random_index.size(); ++i) random_index[i] = i;
    srand(10086);
    random_shuffle(random_index.begin(), random_index.end());


    int PN = 20;
    double lanta = 0.5, lanta_1 = 0.9;
    vector<vector<double>> X(sample_number);
    vector<vector<double>> Y(sample_number);
    vector<vector<double>> edge_index_PN;

    for(int k = 0; k < random_index.size(); ++k){
        int x = random_index[k];

        Y[k] = {node_sample[x][2]};

        int i = node_sample[x][0];
        int j = node_sample[x][1];
        vector<double>ans = met_sim[i];
        ans.insert(ans.end(), dis_sim[j].begin(), dis_sim[j].end());
        X[k] = ans;

        priority_queue<vector<double>, vector<vector<double>>, greater<vector<double>> > q;
        for(int l = 0; l < random_index.size(); ++l){
            int y = random_index[l];

            double MDP_met_sim = met_sim[node_sample[x][0]][node_sample[y][0]];
            double MDP_dis_sim = dis_sim[node_sample[x][1]][node_sample[y][1]];

            double sim = lanta * MDP_met_sim + (1 - lanta) * MDP_dis_sim;

            if(l < PN + 1){
                q.push({sim, k, l});
            }else{
                vector<double>cnt = q.top();
                if(sim - cnt[0] > 1e-6){
                    q.pop();
                    q.push({sim, k, l});
                }
            }
        }
        while(!q.empty()){
            vector<double> cnt = q.top();
            q.pop();
             if(cnt[0] - lanta_1 > 1e-5){
                edge_index_PN.push_back({cnt[2], cnt[1]});
             }
        }
    }

    cout << "X: " << X.size() << ' ' << X[0].size() << '\n';
    cout << "Y: " << Y.size() << ' ' << Y[0].size() << '\n';
    cout << "edge_index_PN len: " << edge_index_PN.size() << ' ' << edge_index_PN[0].size() << '\n';

    Writecsv("X.csv", X);
    Writecsv("Y.csv", Y);
    Writecsv("NN3_1_1_balance_edge_index.csv", edge_index_PN);
}

int main(){
    MDPGTConstruction();
    return 0;
}
