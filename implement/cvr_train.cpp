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

class CoreVectorRegression
{
public:
    double C, mu, gamma, epsilon;
    int max_iter;
    double b, epsilon_bar;
    vector<int> core_idx;
    vector<double> alpha;
    vector<vector<double>> core_cache;
    vector<double> diag_all, delta_all;

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

    void solve_smo(int k, double reg_diag, const vector<double> &d_S,
                   vector<double> &g, double tol,
                   const vector<vector<double>> &K_core)
    {
        int n = 2 * k;

        for (int i = 0; i < n; ++i)
            g[i] = -0.5 * d_S[i];

        for (int j = 0; j < n; ++j)
        {
            if (alpha[j] < 1e-12)
                continue;
            int cj = j >> 1, sj = j & 1;
            const double *cj_row = K_core[cj].data();
            double aj = alpha[j];

            for (int ct = 0; ct < k; ++ct)
            {
                // K̃[2ct, j] và K̃[2ct+1, j]
                double kv = cj_row[ct] + 1.0;
                double diag = (ct == cj) ? reg_diag : 0.0;
                if (sj == 0)
                {
                    g[2 * ct] += (kv + diag) * aj;
                    g[2 * ct + 1] += (-kv) * aj;
                }
                else
                {
                    g[2 * ct] += (-kv) * aj;
                    g[2 * ct + 1] += (kv + diag) * aj;
                }
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
            int ci = ii >> 1;
            const double *ci_row = K_core[ci].data();
            double kii_ws = ci_row[ci] + 1.0 + reg_diag;
            for (int t = 0; t < n; ++t)
            {
                if (max_g - g[t] > 0)
                {
                    int ct = t >> 1;
                    double ktt = K_core[ct][ct] + 1.0 + reg_diag;
                    double kit_base = ci_row[ct] + 1.0;
                    double kit = kit_base;
                    if ((ii & 1) != (t & 1))
                        kit = -kit_base;
                    else if (ci == ct)
                        kit += reg_diag;

                    double quad_coef = kii_ws + ktt - 2.0 * kit;
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

            int cj = ji >> 1, sj = ji & 1;
            int si = ii & 1;

            double kii = K_core[ci][ci] + 1.0 + reg_diag;
            double kjj = K_core[cj][cj] + 1.0 + reg_diag;
            double kij_base = ci_row[cj] + 1.0;
            double kij;
            if (si == sj)
            {
                kij = kij_base + (ci == cj ? reg_diag : 0.0);
            }
            else
            {
                kij = -kij_base;
            }

            double eta = kii + kjj - 2.0 * kij;
            if (eta < 1e-12)
                eta = 1e-12;
            double lam = min((max_g - g[ji]) / eta, alpha[ii]);
            if (lam < 1e-12)
                break;

            alpha[ii] -= lam;
            alpha[ji] += lam;

            const double *cj_row2 = K_core[cj].data();
            for (int ct = 0; ct < k; ++ct)
            {
                double kvi = ci_row[ct] + 1.0;
                double kvj = cj_row2[ct] + 1.0;
                double di = (ct == ci) ? reg_diag : 0.0;
                double dj = (ct == cj) ? reg_diag : 0.0;

                double Ki_pos = (si == 0) ? (kvi + di) : (-kvi);
                double Ki_neg = (si == 0) ? (-kvi) : (kvi + di);
                double Kj_pos = (sj == 0) ? (kvj + dj) : (-kvj);
                double Kj_neg = (sj == 0) ? (-kvj) : (kvj + dj);

                g[2 * ct] += lam * (Kj_pos - Ki_pos);
                g[2 * ct + 1] += lam * (Kj_neg - Ki_neg);
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
    CoreVectorRegression(double C_, double mu_, double gamma_,
                         double eps_, int max_iter_)
        : C(C_), mu(mu_), gamma(gamma_), epsilon(eps_),
          max_iter(max_iter_), b(0), epsilon_bar(0) {}

    void fit(const vector<vector<double>> &X, const vector<double> &y)
    {
        int m = (int)X.size();

        diag_all.assign(2 * m, 0.0);
        delta_all.assign(2 * m, 0.0);
        vector<double> lin_all(2 * m);
        double reg_diag = mu * m / C;
        double min_lin = 1e18, max_diag = -1e18;

        for (int i = 0; i < m; ++i)
        {
            double dv = 2.0 + reg_diag;
            diag_all[i] = diag_all[i + m] = dv;
            lin_all[i] = (2.0 / C) * y[i];
            lin_all[i + m] = -(2.0 / C) * y[i];
            min_lin = min({min_lin, lin_all[i], lin_all[i + m]});
            max_diag = max(max_diag, dv);
        }

        double eta_global = max_diag - min_lin + 1.0;
        for (int i = 0; i < 2 * m; ++i)
            delta_all[i] = max(-diag_all[i] + eta_global + lin_all[i], 0.0);

        mt19937 rng(42);
        vector<int> pool(m);
        iota(pool.begin(), pool.end(), 0);
        shuffle(pool.begin(), pool.end(), rng);
        core_idx = {pool[0], pool[1]};
        unordered_set<int> core_set(core_idx.begin(), core_idx.end());

        for (int c : core_idx)
        {
            vector<double> row(m);
            for (int i = 0; i < m; ++i)
                row[i] = rbf(X[c], X[i]);
            core_cache.push_back(row);
        }
        alpha.assign(4, 0.25);

        vector<vector<double>> K_core(2, vector<double>(2));
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                K_core[i][j] = core_cache[i][core_idx[j]];
            }
        }

        vector<double> g;

        for (int outer = 0; outer < max_iter; ++outer)
        {
            int k = (int)core_idx.size();
            int n = 2 * k;

            vector<double> d_S(n);
            for (int i = 0; i < k; ++i)
            {
                int gi = core_idx[i];
                double sk = core_cache[i][gi] + 1.0 + reg_diag;
                d_S[2 * i] = sk + delta_all[gi] - eta_global;
                d_S[2 * i + 1] = sk + delta_all[gi + m] - eta_global;
            }

            g.assign(n, 0.0);
            solve_smo(k, reg_diag, d_S, g, 1e-6, K_core);

            double sum_ag = 0.0, sum_adS = 0.0;
            for (int i = 0; i < n; ++i)
            {
                sum_ag += alpha[i] * g[i];
                sum_adS += alpha[i] * d_S[i];
            }
            double quad_term = sum_ag + 0.5 * sum_adS;

            b = 0.0;
            for (int i = 0; i < k; ++i)
                b += C * (alpha[2 * i] - alpha[2 * i + 1]);

            double linear_term = 0.0;
            for (int i = 0; i < n; ++i)
            {
                int ci = i >> 1, si = i & 1;
                linear_term += alpha[i] * (si == 0 ? y[core_idx[ci]] : -y[core_idx[ci]]);
            }
            epsilon_bar = max(linear_term - C * quad_term, 0.0);

            double c_norm_sq = quad_term;

            double R2 = max(0.5 * sum_adS + eta_global - sum_ag, 1e-12);
            double threshold = pow((1.0 + epsilon) * sqrt(R2), 2);

            double sum_diff = 0.0;
            for (int j = 0; j < k; ++j)
                sum_diff += alpha[2 * j] - alpha[2 * j + 1];

            auto get_kt = [&](int idx)
            {
                double v = sum_diff;
                for (int j = 0; j < k; ++j)
                {
                    double ad = alpha[2 * j] - alpha[2 * j + 1];
                    if (abs(ad) > 1e-12)
                        v += core_cache[j][idx] * ad;
                }
                return v;
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
                double bd = -1;
                int bi = -1;
                for (int s = 0; s < ss; ++s)
                {
                    int idx = non_core[s];
                    double v = get_kt(idx);
                    double d0 = c_norm_sq - 2.0 * v + diag_all[idx] + delta_all[idx];
                    double d1 = c_norm_sq + 2.0 * v + diag_all[idx + m] + delta_all[idx + m];
                    double d = max(d0, d1);
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
                    alpha.push_back(0.0);

                    int new_k = core_idx.size();
                    K_core.push_back(vector<double>(new_k));
                    for (int i = 0; i < new_k - 1; ++i)
                    {
                        double val = core_cache[new_k - 1][core_idx[i]];
                        K_core[new_k - 1][i] = val;
                        K_core[i].push_back(val);
                    }
                    K_core[new_k - 1][new_k - 1] = core_cache[new_k - 1][bi];

                    added = true;
                }
            }
            if (added)
                continue;

            double bd_all = -1;
            int bi_all = -1;
            for (int idx = 0; idx < m; ++idx)
            {
                if (core_set.count(idx))
                    continue;
                double v = get_kt(idx);
                double d0 = c_norm_sq - 2.0 * v + diag_all[idx] + delta_all[idx];
                double d1 = c_norm_sq + 2.0 * v + diag_all[idx + m] + delta_all[idx + m];
                double d = max(d0, d1);
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
            alpha.push_back(0.0);

            int new_k = core_idx.size();
            K_core.push_back(vector<double>(new_k));
            for (int i = 0; i < new_k - 1; ++i)
            {
                double val = core_cache[new_k - 1][core_idx[i]];
                K_core[new_k - 1][i] = val;
                K_core[i].push_back(val);
            }
            K_core[new_k - 1][new_k - 1] = core_cache[new_k - 1][bi_all];
        }
    }
};

bool load_libsvm(const string &path, int nf,
                 vector<vector<double>> &X, vector<double> &y)
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
    if (argc < 8)
        return 1;
    string train_path = argv[1];
    int nf = stoi(argv[2]);
    double C = stod(argv[3]);
    double mu = stod(argv[4]);
    double gamma = stod(argv[5]);
    double epsilon = stod(argv[6]);
    string out_path = argv[7];

    vector<vector<double>> X;
    vector<double> y;
    load_libsvm(train_path, nf, X, y);

    CoreVectorRegression model(C, mu, gamma, epsilon, 100000);
    model.fit(X, y);

    ofstream out(out_path);
    out << setprecision(10)
        << model.b << "\n"
        << model.epsilon_bar << "\n"
        << model.core_idx.size() << "\n";
    for (size_t i = 0; i < model.core_idx.size(); ++i)
        out << model.core_idx[i] << " "
            << C * (model.alpha[2 * i] - model.alpha[2 * i + 1]) << "\n";
    out.close();
    return 0;
}
