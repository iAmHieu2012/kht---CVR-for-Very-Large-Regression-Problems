#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
using namespace std;

class CoreVectorMachine
{
public:
    double C, gamma, epsilon;
    int max_iter;
    vector<int> core_idx;
    vector<double> alpha;
    vector<vector<double>> core_cache;

private:
    inline double rbf(const vector<double> &x1, const vector<double> &x2) const
    {
        double sq = 0;
        const double *p1 = x1.data();
        const double *p2 = x2.data();
        for (int i = 0, n = (int)x1.size(); i < n; ++i)
        {
            double d = p1[i] - p2[i];
            sq += d * d;
        }
        return exp(-gamma * sq);
    }

    void solve_smo(int k, double reg_diag, vector<double> &g, double tol,
                   const vector<vector<double>> &K_core)
    {
        int n = k;
        for (int i = 0; i < n; ++i)
        {
            g[i] = 0;
            for (int j = 0; j < n; ++j)
            {
                if (alpha[j] > 1e-12)
                    g[i] += alpha[j] * K_core[i][j];
            }
        }

        for (int iter = 0; iter < 2000000; ++iter)
        {
            int ii = -1, ji_min = -1;
            double max_g = -1e18, min_g = 1e18;
            for (int t = 0; t < n; ++t)
            {
                if (alpha[t] > 1e-12 && g[t] > max_g)
                {
                    max_g = g[t];
                    ii = t;
                }
                if (g[t] < min_g)
                {
                    min_g = g[t];
                    ji_min = t;
                }
            }
            if (ii < 0 || ji_min < 0 || max_g - min_g < tol)
                break;

            int ji = -1;
            double max_obj = -1e18;
            double kii = K_core[ii][ii];
            for (int t = 0; t < n; ++t)
            {
                if (max_g - g[t] > 0)
                {
                    double ktt = K_core[t][t];
                    double kit = K_core[ii][t];
                    double quad_coef = kii + ktt - 2.0 * kit;
                    if (quad_coef <= 1e-12)
                        quad_coef = 1e-12;
                    double obj = (max_g - g[t]) * (max_g - g[t]) / quad_coef;
                    if (obj > max_obj)
                    {
                        max_obj = obj;
                        ji = t;
                    }
                }
            }
            if (ji < 0)
                ji = ji_min;

            double kjj = K_core[ji][ji];
            double kij = K_core[ii][ji];
            double eta = kii + kjj - 2.0 * kij;
            if (eta < 1e-12)
                eta = 1e-12;
            double lam = min((max_g - g[ji]) / eta, alpha[ii]);
            if (lam < 1e-12)
                break;

            alpha[ii] -= lam;
            alpha[ji] += lam;

            for (int t = 0; t < n; ++t)
            {
                g[t] += lam * (K_core[t][ji] - K_core[t][ii]);
            }
        }

        double sum = 0.0;
        for (double &a : alpha)
        {
            a = max(0.0, a);
            sum += a;
        }
        if (sum > 0)
            for (double &a : alpha)
                a /= sum;
    }

public:
    CoreVectorMachine(double C_, double gamma_, double eps_, int max_iter_)
        : C(C_), gamma(gamma_), epsilon(eps_), max_iter(max_iter_) {}

    void fit(const vector<vector<double>> &X, const vector<double> &y)
    {
        int m = (int)X.size();
        double reg_diag = 1.0 / C;
        double c_diag = 2.0 + reg_diag;

        mt19937 rng(42);
        vector<int> pool(m);
        iota(pool.begin(), pool.end(), 0);
        shuffle(pool.begin(), pool.end(), rng);

        int pos_idx = -1, neg_idx = -1;
        for (int i = 0; i < m; ++i)
        {
            if (pos_idx < 0 && y[pool[i]] > 0)
                pos_idx = pool[i];
            if (neg_idx < 0 && y[pool[i]] < 0)
                neg_idx = pool[i];
            if (pos_idx >= 0 && neg_idx >= 0)
                break;
        }
        if (pos_idx < 0)
            pos_idx = pool[0];
        if (neg_idx < 0)
            neg_idx = pool[1];

        core_idx = {pos_idx, neg_idx};
        unordered_set<int> core_set(core_idx.begin(), core_idx.end());

        for (int c : core_idx)
        {
            vector<double> row(m);
            for (int i = 0; i < m; ++i)
                row[i] = rbf(X[c], X[i]);
            core_cache.push_back(row);
        }
        alpha.assign(2, 0.5);

        vector<vector<double>> K_core(2, vector<double>(2));
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                double val = y[core_idx[i]] * y[core_idx[j]] * (core_cache[i][core_idx[j]] + 1.0);
                if (i == j)
                    val += reg_diag;
                K_core[i][j] = val;
            }
        }

        vector<double> g;

        for (int outer = 0; outer < max_iter; ++outer)
        {
            int k = (int)core_idx.size();
            g.assign(k, 0.0);
            solve_smo(k, reg_diag, g, 1e-6, K_core);

            double quad_term = 0.0;
            for (int i = 0; i < k; ++i)
                quad_term += alpha[i] * g[i];

            double R2 = max(c_diag - quad_term, 1e-12);
            double threshold = pow((1.0 + epsilon) * sqrt(R2), 2);

            double sum_ay = 0.0;
            for (int j = 0; j < k; ++j)
                sum_ay += alpha[j] * y[core_idx[j]];

            auto get_v = [&](int idx)
            {
                double v = sum_ay;
                for (int j = 0; j < k; ++j)
                {
                    if (alpha[j] > 1e-12)
                    {
                        v += alpha[j] * y[core_idx[j]] * core_cache[j][idx];
                    }
                }
                return y[idx] * v;
            };

            vector<int> non_core;
            non_core.reserve(m - k);
            for (int i = 0; i < m; ++i)
                if (!core_set.count(i))
                    non_core.push_back(i);

            bool added = false;
            int ss = min(59, (int)non_core.size());
            if (ss > 0)
            {
                shuffle(non_core.begin(), non_core.end(), rng);
                double bd = -1e18;
                int bi = -1;
                for (int s = 0; s < ss; ++s)
                {
                    int idx = non_core[s];
                    double v = get_v(idx);
                    double d = c_diag - 2.0 * v + quad_term;
                    if (d > threshold && d > bd)
                    {
                        bd = d;
                        bi = idx;
                    }
                }
                if (bi >= 0)
                {
                    core_idx.push_back(bi);
                    core_set.insert(bi);
                    vector<double> row(m);
                    for (int i = 0; i < m; ++i)
                        row[i] = rbf(X[bi], X[i]);
                    core_cache.push_back(row);
                    alpha.push_back(0.0);

                    int new_k = core_idx.size();
                    K_core.push_back(vector<double>(new_k));
                    for (int i = 0; i < new_k - 1; ++i)
                    {
                        double val = y[bi] * y[core_idx[i]] * (core_cache[new_k - 1][core_idx[i]] + 1.0);
                        K_core[new_k - 1][i] = val;
                        K_core[i].push_back(val);
                    }
                    K_core[new_k - 1][new_k - 1] = c_diag;
                    added = true;
                }
            }
            if (added)
                continue;

            double bd_all = -1e18;
            int bi_all = -1;
            for (int idx = 0; idx < m; ++idx)
            {
                if (core_set.count(idx))
                    continue;
                double v = get_v(idx);
                double d = c_diag - 2.0 * v + quad_term;
                if (d > threshold && d > bd_all)
                {
                    bd_all = d;
                    bi_all = idx;
                }
            }

            if (bi_all < 0)
                break;

            core_idx.push_back(bi_all);
            core_set.insert(bi_all);
            vector<double> row(m);
            for (int i = 0; i < m; ++i)
                row[i] = rbf(X[bi_all], X[i]);
            core_cache.push_back(row);
            alpha.push_back(0.0);

            int new_k = core_idx.size();
            K_core.push_back(vector<double>(new_k));
            for (int i = 0; i < new_k - 1; ++i)
            {
                double val = y[bi_all] * y[core_idx[i]] * (core_cache[new_k - 1][core_idx[i]] + 1.0);
                K_core[new_k - 1][i] = val;
                K_core[i].push_back(val);
            }
            K_core[new_k - 1][new_k - 1] = c_diag;
        }

        double quad_term = 0.0;
        for (int i = 0; i < alpha.size(); ++i)
            quad_term += alpha[i] * g[i];

        for (int i = 0; i < alpha.size(); ++i)
        {
            alpha[i] = alpha[i] / quad_term * y[core_idx[i]];
        }
    }
};

bool load_libsvm(const string &path, int nf, vector<vector<double>> &X, vector<double> &y)
{
    ifstream f(path);
    if (!f.is_open())
        return false;
    string line;
    while (getline(f, line))
    {
        if (line.empty())
            continue;
        stringstream ss(line);
        double lbl;
        ss >> lbl;
        y.push_back(lbl);
        vector<double> x(nf, 0.0);
        string tok;
        while (ss >> tok)
        {
            auto p = tok.find(':');
            if (p != string::npos)
            {
                int idx = stoi(tok.substr(0, p));
                if (idx >= 1 && idx <= nf)
                    x[idx - 1] = stod(tok.substr(p + 1));
            }
        }
        X.push_back(x);
    }
    return true;
}

int main(int argc, char *argv[])
{
    if (argc < 7)
        return 1;
    string train_path = argv[1];
    int nf = stoi(argv[2]);
    double C = stod(argv[3]);
    double gamma = stod(argv[4]);
    double epsilon = stod(argv[5]);
    string out_path = argv[6];

    vector<vector<double>> X;
    vector<double> y;
    load_libsvm(train_path, nf, X, y);

    CoreVectorMachine model(C, gamma, epsilon, 100000);
    model.fit(X, y);

    double b = 0.0;
    vector<pair<int, double>> svs;
    for (int i = 0; i < model.alpha.size(); ++i)
    {
        if (abs(model.alpha[i]) > 1e-12)
        {
            svs.push_back({model.core_idx[i], model.alpha[i]});
            b += model.alpha[i];
        }
    }

    ofstream out(out_path);
    out << setprecision(10) << b << "\n"
        << "0.0\n"
        << svs.size() << "\n";
    for (auto &sv : svs)
    {
        out << sv.first << " " << sv.second << "\n";
    }
    return 0;
}
