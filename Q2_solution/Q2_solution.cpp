
// Question 2 solution
// I have kept the code split into small parts because the problem has many steps.
// Input.csv is read from the same folder and the final answer goes into question2_output.csv.
// Most rates are stored as decimals inside the code, not as percentages.


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <functional>
#include <limits>

namespace {

// These constants are used throughout the file.
// The question follows a 360-day year, and swaps pay every 180 days.
constexpr double DAYS_PER_YEAR  = 360.0;  // day count used for year fraction
constexpr double SWAP_PERIOD    = 180.0;  // semi-annual swap payments
constexpr double NEW_SWAP_NTL   = 100.0;  // notional given in question
constexpr double TIME_EPS       = 1e-9;  // tiny buffer for comparing times
constexpr double PCT_PER_UNIT   = 0.01;  // used when risk is for 1 percent move

enum class Interp { Linear, AvgQuad };

double dcf(double t2, double t1) { return (t2 - t1) / DAYS_PER_YEAR; }

// Removes extra spaces/newlines around a csv field.
std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    auto is_ws = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    while (a < b && is_ws(s[a])) ++a;
    while (b > a && is_ws(s[b - 1])) --b;
    return s.substr(a, b - a);
}

// Splits one csv line into fields. This is enough here because input is simple.
std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) out.push_back(trim(field));
    return out;
}

// Some csv files have empty lines, so I skip those safely.
bool is_blank(const std::string& line) {
    for (char c : line)
        if (c != ',' && c != ' ' && c != '\t' && c != '\r' && c != '\n') return false;
    return true;
}

// Converts maturity like 6M or 2Y into days.
// I am using the assignment convention: 1M = 30 days and 1Y = 360 days.
int parse_maturity(const std::string& raw) {
    std::string s = trim(raw);
    if (s.size() < 2) throw std::runtime_error("bad maturity token: '" + raw + "'");
    char unit = s.back();
    int n;
    try {
        n = std::stoi(s.substr(0, s.size() - 1));
    } catch (...) {
        throw std::runtime_error("bad maturity number: '" + raw + "'");
    }
    switch (unit) {
        case 'D': case 'd': return n;
        case 'W': case 'w': return n * 7;
        case 'M': case 'm': return n * 30;
        case 'Y': case 'y': return n * 360;
        default: throw std::runtime_error("unknown maturity unit: '" + raw + "'");
    }
}

// Converts payment frequency like 6m into days.
int parse_frequency(const std::string& raw) {
    std::string s = trim(raw);
    if (s.empty()) throw std::runtime_error("empty frequency token");
    if (s.back() == 'm' || s.back() == 'M') s.pop_back();
    int months;
    try {
        months = std::stoi(s);
    } catch (...) {
        throw std::runtime_error("bad frequency token: '" + raw + "'");
    }
    if (months <= 0) throw std::runtime_error("non-positive frequency: '" + raw + "'");
    return months * 30;  // rough month count
}

// This struct keeps all input values together so functions dont need too many arguments.
struct Input {
    int n = 0;
    std::vector<double> maturity;
    std::vector<double> cash_rate;
    std::vector<double> swap_rate;
    double query_t = 0.0;

    double new_fixed_rate = 0.0;
    double new_maturity   = 0.0;
    double new_fixed_freq = 0.0;
    double new_float_freq = 0.0;
};

// Reads the input csv in the exact order given in the problem.
// First row has number of instruments, then rate rows, then query and new swap details.
Input read_input(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("cannot open input file: " + path);

    std::vector<std::string> lines;
    std::string line;
    bool first = true;
    while (std::getline(fin, line)) {
        // Sometimes csv saved from Excel starts with a BOM. This removes it if present.
        if (first) {
            if (line.size() >= 3 &&
                static_cast<unsigned char>(line[0]) == 0xEF &&
                static_cast<unsigned char>(line[1]) == 0xBB &&
                static_cast<unsigned char>(line[2]) == 0xBF) {
                line = line.substr(3);
            }
            first = false;
        }
        if (!is_blank(line)) lines.push_back(line);
    }

    if (lines.empty()) throw std::runtime_error("input file is empty");

    Input in;
    size_t row = 0;

    auto head = split(lines.at(row++));
    if (head.empty()) throw std::runtime_error("missing instrument count");
    in.n = std::stoi(head[0]);
    if (in.n <= 0) throw std::runtime_error("instrument count must be positive");

    for (int i = 0; i < in.n; ++i) {
        if (row >= lines.size()) throw std::runtime_error("not enough rate rows");
        auto f = split(lines.at(row++));
        if (f.size() < 3) throw std::runtime_error("rate row needs 3 columns");
        in.maturity.push_back(parse_maturity(f[0]));
        in.cash_rate.push_back(std::stod(f[1]) / 100.0);
        in.swap_rate.push_back(std::stod(f[2]) / 100.0);
    }

    // Curve construction assumes maturities come in increasing order.
    for (int i = 1; i < in.n; ++i)
        if (in.maturity[i] <= in.maturity[i - 1])
            throw std::runtime_error("maturities must be strictly increasing");

    if (row >= lines.size()) throw std::runtime_error("missing query time row");
    auto q = split(lines.at(row++));
    if (q.empty()) throw std::runtime_error("empty query time row");
    in.query_t = std::stod(q[0]);

    if (row >= lines.size()) throw std::runtime_error("missing new swap row");
    auto s = split(lines.at(row++));
    if (s.size() < 4) throw std::runtime_error("new swap row needs 4 columns");
    in.new_fixed_rate = std::stod(s[0]) / 100.0;
    in.new_maturity   = parse_maturity(s[1]);
    in.new_fixed_freq = parse_frequency(s[2]);
    in.new_float_freq = parse_frequency(s[3]);  // kept from input

    return in;
}

// One curve point: time and log(discount factor).
// Using log DF makes interpolation smoother and avoids negative DFs.
struct Node { double t; double log_df; };
using Curve = std::vector<Node>;

struct Lag3 { double l0, l1, l2; };

// Lagrange weights for three nearby curve nodes.
// These weights tell how much each node contributes to the quadratic estimate.
Lag3 lagrange3(const Curve& c, int i0, int i1, int i2, double t) {
    const double x0 = c[i0].t, x1 = c[i1].t, x2 = c[i2].t;
    Lag3 w;
    w.l0 = ((t - x1) * (t - x2)) / ((x0 - x1) * (x0 - x2));
    w.l1 = ((t - x0) * (t - x2)) / ((x1 - x0) * (x1 - x2));
    w.l2 = ((t - x0) * (t - x1)) / ((x2 - x0) * (x2 - x1));
    return w;
}

// Interpolates log DF using 3 points.
double quad_log(const Curve& c, int i0, int i1, int i2, double t) {
    const Lag3 w = lagrange3(c, i0, i1, i2, t);
    return w.l0 * c[i0].log_df + w.l1 * c[i1].log_df + w.l2 * c[i2].log_df;
}

// Finds the interval [node i, node i+1] where the required time lies.
int find_interval(const Curve& c, double t) {
    const int n = static_cast<int>(c.size());
    int a = n - 2;
    for (int i = 0; i + 1 < n; ++i) {
        if (t <= c[i + 1].t + TIME_EPS) { a = i; break; }
    }
    return a;
}

// Main interpolation function for log discount factors.
// Linear is direct between two nodes. AvgQuad mixes left and right quadratic fits when possible.
double interp_log_df(const Curve& c, double t, Interp method) {
    const int n = static_cast<int>(c.size());
    if (n == 0) throw std::runtime_error("interpolation on empty curve");
    if (n == 1) return c[0].log_df;

    if (t < c.front().t - TIME_EPS || t > c.back().t + TIME_EPS)
        throw std::runtime_error("interpolation time out of curve range");

    const int a = find_interval(c, t);

    if (t <= c[a].t + TIME_EPS)     return c[a].log_df;
    if (t >= c[a + 1].t - TIME_EPS) return c[a + 1].log_df;

    const double ta = c[a].t, tb = c[a + 1].t;
    const double lam = (t - ta) / (tb - ta);

    // At the first interval, quadratic needs a point on the left, so linear is safer.
    if (method == Interp::Linear || a == 0) {
        return (1.0 - lam) * c[a].log_df + lam * c[a + 1].log_df;
    }

    const double wa = (tb - t) / (tb - ta);
    const double wb = (t - ta) / (tb - ta);
    const double q_left = quad_log(c, a - 1, a, a + 1, t);
    if (a + 2 <= n - 1) {
        const double q_right = quad_log(c, a, a + 1, a + 2, t);
        return wa * q_left + wb * q_right;
    }
    return q_left;  // last interval has no right-side quadratic
}

// Converts interpolated log DF back to normal discount factor.
double interp_df(const Curve& c, double t, Interp method) {
    return std::exp(interp_log_df(c, t, method));
}

// Same interpolation logic as above, but returns weights for each node.
// This is needed later for risk / sensitivity calculation.
std::vector<double> log_df_weights(const Curve& c, double t, Interp method) {
    const int n = static_cast<int>(c.size());
    std::vector<double> w(static_cast<size_t>(n), 0.0);
    if (n == 0) throw std::runtime_error("weights on empty curve");
    if (n == 1) { w[0] = 1.0; return w; }

    if (t < c.front().t - TIME_EPS || t > c.back().t + TIME_EPS)
        throw std::runtime_error("weight time out of curve range");

    const int a = find_interval(c, t);

    if (t <= c[a].t + TIME_EPS)     { w[a] = 1.0;     return w; }
    if (t >= c[a + 1].t - TIME_EPS) { w[a + 1] = 1.0; return w; }

    const double ta = c[a].t, tb = c[a + 1].t;
    const double lam = (t - ta) / (tb - ta);

    // At the first interval, quadratic needs a point on the left, so linear is safer.
    if (method == Interp::Linear || a == 0) {
        w[a]     = 1.0 - lam;
        w[a + 1] = lam;
        return w;
    }

    const double wa = (tb - t) / (tb - ta);
    const double wb = (t - ta) / (tb - ta);
    const Lag3 left = lagrange3(c, a - 1, a, a + 1, t);
    if (a + 2 <= n - 1) {
        const Lag3 right = lagrange3(c, a, a + 1, a + 2, t);
        w[a - 1] += wa * left.l0;
        w[a]     += wa * left.l1 + wb * right.l0;
        w[a + 1] += wa * left.l2 + wb * right.l1;
        w[a + 2] += wb * right.l2;
    } else {
        w[a - 1] += left.l0;
        w[a]     += left.l1;
        w[a + 1] += left.l2;
    }
    return w;
}

// Derivative of interpolated DF with respect to each curve node DF.
// Basically chain rule because we interpolate log(DF), not DF directly.
std::vector<double> d_interp_df_d_node_df(const Curve& c, double t, Interp method) {
    const std::vector<double> w = log_df_weights(c, t, method);
    const double df_t = interp_df(c, t, method);
    const int n = static_cast<int>(c.size());
    std::vector<double> d(static_cast<size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
        if (w[j] == 0.0) continue;
        const double df_j = std::exp(c[j].log_df);
        d[j] = df_t * w[j] / df_j;
    }
    return d;
}

struct Payment { double t; double dcf; };

// Builds swap payment dates from today to maturity.
// Also stores accrual fraction for each period.
std::vector<Payment> make_schedule(double maturity, double period) {
    std::vector<Payment> sched;
    double prev = 0.0, cur = period;
    while (cur < maturity - TIME_EPS) {
        sched.push_back({cur, dcf(cur, prev)});
        prev = cur;
        cur += period;
    }
    sched.push_back({maturity, dcf(maturity, prev)});  // last payment
    return sched;
}

// Simple bisection root solver.
// I use this while bootstrapping because the unknown DF is solved one node at a time.
double solve_decreasing(const std::function<double(double)>& f,
                        double lo, double hi) {
    double flo = f(lo), fhi = f(hi);
    if (flo * fhi > 0.0)
        throw std::runtime_error("curve calibration failed to bracket a root");

    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double fmid = f(mid);
        if (std::fabs(fmid) < 1e-14 || (hi - lo) < 1e-14) return mid;
        if (flo * fmid <= 0.0) { hi = mid; fhi = fmid; }
        else                   { lo = mid; flo = fmid; }
    }
    return 0.5 * (lo + hi);
}

// Builds the cash curve directly from simple money-market/cash rates.
// Formula used: DF = 1 / (1 + r * tau).
Curve build_cash_curve(const Input& in) {
    Curve c;
    c.reserve(in.n);
    for (int i = 0; i < in.n; ++i) {
        const double T = in.maturity[i];
        const double df = 1.0 / (1.0 + in.cash_rate[i] * dcf(T, 0.0));
        if (df <= 0.0) throw std::runtime_error("non-positive cash discount factor");
        c.push_back({T, std::log(df)});
    }
    return c;
}

// Builds the swap curve by bootstrapping.
// For each maturity, previous nodes are already known and current node is solved.
Curve build_swap_curve(const Input& in, Interp method) {
    Curve c;
    c.reserve(in.n);

    for (int j = 0; j < in.n; ++j) {
        const double T = in.maturity[j];
        const double p = in.swap_rate[j];

        // If maturity is before first swap payment date, treat it like a simple cash-style quote.
        if (T < SWAP_PERIOD) {
            const double df = 1.0 / (1.0 + p * dcf(T, 0.0));
            c.push_back({T, std::log(df)});
            continue;
        }

        const std::vector<Payment> sched = make_schedule(T, SWAP_PERIOD);
        c.push_back({T, 0.0});  // temporary value
        const int idx = static_cast<int>(c.size()) - 1;

        // Residual is zero when the par swap equation matches the market swap rate.
        auto residual = [&](double x) -> double {
            c[idx].log_df = std::log(x);
            double annuity = 0.0;
            for (const Payment& pay : sched)
                annuity += pay.dcf * interp_df(c, pay.t, method);
            return (1.0 - x) - p * annuity;
        };

        const double x = solve_decreasing(residual, 1e-12, 1.0);
        c[idx].log_df = std::log(x);
    }
    return c;
}

// Present value of 1 unit paid on the fixed leg schedule.
double fixed_annuity(const Curve& c, Interp method,
                     double maturity, double fixed_freq) {
    const std::vector<Payment> sched = make_schedule(maturity, fixed_freq);
    double annuity = 0.0;
    for (const Payment& pay : sched)
        annuity += pay.dcf * interp_df(c, pay.t, method);
    return annuity;
}

// PV is from payer-swap view used here: receive floating and pay fixed.
double swap_pv(const Curve& c, Interp method, const Input& in) {
    const double ann = fixed_annuity(c, method, in.new_maturity, in.new_fixed_freq);
    const double df_T = interp_df(c, in.new_maturity, method);
    const double pv_fixed = NEW_SWAP_NTL * in.new_fixed_rate * ann;  // fixed leg
    const double pv_float = NEW_SWAP_NTL * (1.0 - df_T);  // floating leg
    return pv_float - pv_fixed;
}

// Par rate is the fixed rate that makes the new swap PV equal to zero.
double swap_par_rate(const Curve& c, Interp method, const Input& in) {
    const double ann = fixed_annuity(c, method, in.new_maturity, in.new_fixed_freq);
    const double df_T = interp_df(c, in.new_maturity, method);
    if (ann <= 0.0) throw std::runtime_error("non-positive fixed annuity");
    return (1.0 - df_T) / ann;  // in decimal
}

using Matrix = std::vector<std::vector<double>>;

// Jacobian for cash curve.
// Cash instruments are independent, so this matrix is diagonal.
Matrix cash_jacobian(const Input& in, const Curve& cash) {
    const int n = in.n;
    Matrix J(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(n), 0.0));
    for (int j = 0; j < n; ++j) {
        const double df_j = std::exp(cash[j].log_df);
        const double tau  = in.maturity[j] / DAYS_PER_YEAR;
        J[j][j] = -tau * df_j * df_j;
    }
    return J;
}

// Jacobian for swap curve.
// This is more involved because every bootstrapped DF depends on earlier DFs too.
Matrix swap_jacobian(const Input& in, const Curve& full, Interp method) {
    const int n = in.n;
    Matrix J(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(n), 0.0));

    for (int j = 0; j < n; ++j) {
        const double T = in.maturity[j];
        const double p = in.swap_rate[j];

        const Curve partial(full.begin(), full.begin() + (j + 1));  // nodes till current point
        const std::vector<Payment> sched = make_schedule(T, SWAP_PERIOD);

        // G is the bootstrap equation. These store its partial derivatives.
        double dG_dx = -1.0;
        double dG_dp = 0.0;
        std::vector<double> dG_dDFm(static_cast<size_t>(j), 0.0);

        for (const Payment& pay : sched) {
            const double df_ti = interp_df(partial, pay.t, method);
            const std::vector<double> w = d_interp_df_d_node_df(partial, pay.t, method);  // same size as partial curve
            dG_dp -= pay.dcf * df_ti;
            dG_dx -= p * pay.dcf * w[static_cast<size_t>(j)];
            for (int m = 0; m < j; ++m)
                dG_dDFm[static_cast<size_t>(m)] -= p * pay.dcf * w[static_cast<size_t>(m)];
        }

        // Implicit function theorem step: convert quote movement into DF movement.
        for (int k = 0; k < n; ++k) {
            double rhs = (k == j) ? dG_dp : 0.0;
            for (int m = 0; m < j; ++m)
                rhs += dG_dDFm[static_cast<size_t>(m)] * J[m][k];
            J[j][k] = -rhs / dG_dx;
        }
    }
    return J;
}

// Computes risk of the new swap against each original market quote.
// First get PV sensitivity to curve DFs, then pass it through the curve Jacobian.
std::vector<double> swap_risk(const Curve& curve, Interp method,
                              const Input& in, const Matrix& J) {
    const int n = static_cast<int>(curve.size());
    std::vector<double> P(static_cast<size_t>(n), 0.0);  // PV sensitivity to discount factors

    const double N = NEW_SWAP_NTL;
    const double K = in.new_fixed_rate;

    const std::vector<Payment> fix = make_schedule(in.new_maturity, in.new_fixed_freq);
    for (const Payment& pay : fix) {
        const double s_t = -N * K * pay.dcf;
        const std::vector<double> w = d_interp_df_d_node_df(curve, pay.t, method);
        for (int j = 0; j < n; ++j) P[static_cast<size_t>(j)] += s_t * w[static_cast<size_t>(j)];
    }

    // Floating leg also depends on the final maturity discount factor.
    {
        const double s_t = -N;
        const std::vector<double> w = d_interp_df_d_node_df(curve, in.new_maturity, method);
        for (int j = 0; j < n; ++j) P[static_cast<size_t>(j)] += s_t * w[static_cast<size_t>(j)];
    }

    std::vector<double> risk(static_cast<size_t>(n), 0.0);
    for (int k = 0; k < n; ++k) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) sum += P[static_cast<size_t>(j)] * J[j][k];
        risk[static_cast<size_t>(k)] = sum * PCT_PER_UNIT;
    }
    return risk;
}

// Writes output in the required csv format.
// First rows are DF/PV/par rate, then one risk row per market quote.
void write_output(const std::string& path,
                  const double q1[4],
                  const double pv[4],
                  const double par_pct[4],
                  const std::vector<double>& risk_cash_lin,
                  const std::vector<double>& risk_cash_aq,
                  const std::vector<double>& risk_swap_lin,
                  const std::vector<double>& risk_swap_aq) {
    std::ofstream fout(path);
    if (!fout) throw std::runtime_error("cannot open output file: " + path);
    fout << std::setprecision(12);

    auto row = [&](double a, double b, double cc, double d) {
        fout << a << "," << b << "," << cc << "," << d << "\n";
    };

    row(q1[0], q1[1], q1[2], q1[3]);  // Q1 dfs
    row(pv[0], pv[1], pv[2], pv[3]);  // PV row
    row(par_pct[0], par_pct[1], par_pct[2], par_pct[3]);  // par rates

    const size_t m = risk_cash_lin.size();  // number of quotes
    for (size_t k = 0; k < m; ++k)  // risk rows
        row(risk_cash_lin[k], risk_cash_aq[k], risk_swap_lin[k], risk_swap_aq[k]);
}

}

// Main just connects all steps: read input, build curves, calculate outputs, write csv.
int main() {
    try {
        const Input in = read_input("Input.csv");

        // Build all curves needed for the four output columns.
        const Curve cash      = build_cash_curve(in);
        const Curve swap_lin  = build_swap_curve(in, Interp::Linear);
        const Curve swap_aq   = build_swap_curve(in, Interp::AvgQuad);

        // Q1 values: discount factor at the query time for each curve/interpolation combo.
        double q1[4];
        q1[0] = interp_df(cash,     in.query_t, Interp::Linear);
        q1[1] = interp_df(cash,     in.query_t, Interp::AvgQuad);
        q1[2] = interp_df(swap_lin, in.query_t, Interp::Linear);
        q1[3] = interp_df(swap_aq,  in.query_t, Interp::AvgQuad);

        // PV and par rate of the new swap under the same four methods.
        double pv[4], par_pct[4];
        pv[0] = swap_pv(cash,     Interp::Linear,  in);
        pv[1] = swap_pv(cash,     Interp::AvgQuad, in);
        pv[2] = swap_pv(swap_lin, Interp::Linear,  in);
        pv[3] = swap_pv(swap_aq,  Interp::AvgQuad, in);

        par_pct[0] = swap_par_rate(cash,     Interp::Linear,  in) * 100.0;
        par_pct[1] = swap_par_rate(cash,     Interp::AvgQuad, in) * 100.0;
        par_pct[2] = swap_par_rate(swap_lin, Interp::Linear,  in) * 100.0;
        par_pct[3] = swap_par_rate(swap_aq,  Interp::AvgQuad, in) * 100.0;

        // These matrices connect market quote changes to curve DF changes.
        const Matrix J_cash     = cash_jacobian(in, cash);
        const Matrix J_swap_lin = swap_jacobian(in, swap_lin, Interp::Linear);
        const Matrix J_swap_aq  = swap_jacobian(in, swap_aq,  Interp::AvgQuad);

        // Final risk rows for all four approaches.
        const std::vector<double> risk_cash_lin = swap_risk(cash,     Interp::Linear,  in, J_cash);
        const std::vector<double> risk_cash_aq  = swap_risk(cash,     Interp::AvgQuad, in, J_cash);
        const std::vector<double> risk_swap_lin = swap_risk(swap_lin, Interp::Linear,  in, J_swap_lin);
        const std::vector<double> risk_swap_aq  = swap_risk(swap_aq,  Interp::AvgQuad, in, J_swap_aq);

        write_output("question2_output.csv", q1, pv, par_pct,
                     risk_cash_lin, risk_cash_aq, risk_swap_lin, risk_swap_aq);

        std::cout << "Done. Wrote question2_output.csv\n";
        return 0;
    } catch (const std::exception& e) {
        // If input format/path is wrong, show the error instead of crashing silently.
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
