// Fast C++20 Rectangular Box Smooth SAT Penalty L-BFGS + Height Bisection
// Quencher Usage: rect_quench IN_TXT OUT_TXT W_FIXED [MODE: quench|seg_grow]
// [MAX_H_INIT] File format: "m H\nx y theta_deg\n..."
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Sq {
  double x, y, deg;
};

static double eval_energy_grad(const std::vector<double> &v, double W, double H,
                               double wb, std::vector<double> &g) {
  int n = (int)v.size() / 3;
  g.assign(3 * n, 0.0);
  std::vector<double> c(n), s(n);
  for (int i = 0; i < n; ++i) {
    double th = v[2 * n + i];
    c[i] = std::cos(th);
    s[i] = std::sin(th);
  }

  const double cs = 1.42;
  int gx_dim = std::max(3, (int)(W / cs) + 3);
  int gy_dim = std::max(3, (int)(H / cs) + 3);
  const int CAP = 32;
  std::vector<int> cnt((size_t)gx_dim * gy_dim, 0);
  std::vector<int> items((size_t)gx_dim * gy_dim * CAP, -1);
  std::vector<int> gxi(n), gyi(n);

  for (int i = 0; i < n; ++i) {
    int gx = std::min(std::max((int)(v[i] / cs) + 1, 0), gx_dim - 1);
    int gy = std::min(std::max((int)(v[n + i] / cs) + 1, 0), gy_dim - 1);
    int cell = gx * gy_dim + gy;
    if (cnt[cell] < CAP) {
      items[(size_t)cell * CAP + cnt[cell]] = i;
      cnt[cell]++;
    }
    gxi[i] = gx;
    gyi[i] = gy;
  }

  double E = 0.0;
  for (int i = 0; i < n; ++i) {
    double ci = c[i], si = s[i];
    double sgn_c = (ci >= 0.0) ? 1.0 : -1.0;
    double sgn_s = (si >= 0.0) ? 1.0 : -1.0;
    double h = 0.5 * (std::abs(ci) + std::abs(si));
    double dh = 0.5 * (-sgn_c * si + sgn_s * ci);

    // X wall [0, W]
    double px = v[i];
    double lox = h - px;
    if (lox > 0.0) {
      E += wb * lox * lox;
      g[i] += -2.0 * wb * lox;
      g[2 * n + i] += 2.0 * wb * lox * dh;
    }
    double hix = px + h - W;
    if (hix > 0.0) {
      E += wb * hix * hix;
      g[i] += 2.0 * wb * hix;
      g[2 * n + i] += 2.0 * wb * hix * dh;
    }
    // Y wall [0, H]
    double py = v[n + i];
    double loy = h - py;
    if (loy > 0.0) {
      E += wb * loy * loy;
      g[n + i] += -2.0 * wb * loy;
      g[2 * n + i] += 2.0 * wb * loy * dh;
    }
    double hiy = py + h - H;
    if (hiy > 0.0) {
      E += wb * hiy * hiy;
      g[n + i] += 2.0 * wb * hiy;
      g[2 * n + i] += 2.0 * wb * hiy * dh;
    }

    // Pairs in 3x3 neighborhood
    for (int ax_ = std::max(0, gxi[i] - 1);
         ax_ <= std::min(gx_dim - 1, gxi[i] + 1); ++ax_) {
      for (int ay_ = std::max(0, gyi[i] - 1);
           ay_ <= std::min(gy_dim - 1, gyi[i] + 1); ++ay_) {
        int cell = ax_ * gy_dim + ay_;
        int cnum = cnt[cell];
        for (int q = 0; q < cnum; ++q) {
          int j = items[(size_t)cell * CAP + q];
          if (j <= i)
            continue;
          double dx = v[j] - v[i], dy = v[n + j] - v[n + i];
          if (dx * dx + dy * dy >= 2.0)
            continue;
          double best = 1e300;
          int bk = -1;
          double cj = c[j], sj = s[j];
          for (int k = 0; k < 4; ++k) {
            double ax, ay;
            if (k == 0) {
              ax = ci;
              ay = si;
            } else if (k == 1) {
              ax = -si;
              ay = ci;
            } else if (k == 2) {
              ax = cj;
              ay = sj;
            } else {
              ax = -sj;
              ay = cj;
            }
            double ri = 0.5 * (std::abs(ax * ci + ay * si) +
                               std::abs(-ax * si + ay * ci));
            double rj = 0.5 * (std::abs(ax * cj + ay * sj) +
                               std::abs(-ax * sj + ay * cj));
            double o = ri + rj - std::abs(ax * dx + ay * dy);
            if (o <= 0.0) {
              bk = -1;
              break;
            }
            if (o < best) {
              best = o;
              bk = k;
            }
          }
          if (bk < 0)
            continue;
          double o = best;
          E += o * o;
          double w = 2.0 * o;
          int own = (bk < 2) ? i : j;
          int oth = (bk < 2) ? j : i;
          double co = c[own], so = s[own];
          double ax, ay, dax, day;
          if (bk % 2 == 0) {
            ax = co;
            ay = so;
            dax = -so;
            day = co;
          } else {
            ax = -so;
            ay = co;
            dax = -co;
            day = -so;
          }
          double proj = ax * dx + ay * dy;
          double sg = (proj >= 0.0) ? 1.0 : -1.0;
          g[j] += w * (-sg * ax);
          g[n + j] += w * (-sg * ay);
          g[i] += w * (sg * ax);
          g[n + i] += w * (sg * ay);

          double cot = c[oth], sot = s[oth];
          double p1 = ax * cot + ay * sot;
          double p2 = -ax * sot + ay * cot;
          double s1 = (p1 >= 0.0) ? 1.0 : -1.0;
          double s2 = (p2 >= 0.0) ? 1.0 : -1.0;
          double dp1_dto = -ax * sot + ay * cot;
          double dp2_dto = -ax * cot - ay * sot;
          g[2 * n + oth] += w * 0.5 * (s1 * dp1_dto + s2 * dp2_dto);

          double dp1_da = dax * cot + day * sot;
          double dp2_da = -dax * sot + day * cot;
          double dproj = dax * dx + day * dy;
          g[2 * n + own] +=
              w * (0.5 * (s1 * dp1_da + s2 * dp2_da) - sg * dproj);
        }
      }
    }
  }
  return E;
}

static double lbfgs_minimize(std::vector<double> &v, double W, double H,
                             int max_iter, double tolE) {
  int N = (int)v.size();
  const int m = 12;
  std::vector<std::vector<double>> sk(m, std::vector<double>(N, 0.0));
  std::vector<std::vector<double>> yk(m, std::vector<double>(N, 0.0));
  std::vector<double> rho(m, 0.0), alpha_hist(m, 0.0);
  std::vector<double> g, gn, q(N), r(N), d(N), vn(N), s_(N), y_(N);

  double E = eval_energy_grad(v, W, H, 1.0, g);
  int k = 0, stall = 0;
  for (int it = 0; it < max_iter; ++it) {
    if (E < tolE)
      return E;
    q = g;
    int cnt = std::min(k, m);
    for (int jj = 0; jj < cnt; ++jj) {
      int j = (k - 1 - jj) % m;
      double dot = 0.0;
      for (int t = 0; t < N; ++t)
        dot += sk[j][t] * q[t];
      double a = rho[j] * dot;
      alpha_hist[j] = a;
      for (int t = 0; t < N; ++t)
        q[t] -= a * yk[j][t];
    }
    double gamma;
    if (cnt > 0) {
      int j = (k - 1) % m;
      double sy = 0.0, yy = 0.0;
      for (int t = 0; t < N; ++t) {
        sy += sk[j][t] * yk[j][t];
        yy += yk[j][t] * yk[j][t];
      }
      gamma = sy / std::max(yy, 1e-300);
    } else {
      double gg = 0.0;
      for (int t = 0; t < N; ++t)
        gg += g[t] * g[t];
      gamma = 1e-3 / std::max(std::sqrt(gg), 1e-300);
    }
    for (int t = 0; t < N; ++t)
      r[t] = gamma * q[t];
    for (int jj = cnt - 1; jj >= 0; --jj) {
      int j = (k - 1 - jj) % m;
      double dot = 0.0;
      for (int t = 0; t < N; ++t)
        dot += yk[j][t] * r[t];
      double b = rho[j] * dot;
      double diff = alpha_hist[j] - b;
      for (int t = 0; t < N; ++t)
        r[t] += sk[j][t] * diff;
    }
    double gd = 0.0;
    for (int t = 0; t < N; ++t) {
      d[t] = -r[t];
      gd += g[t] * d[t];
    }
    if (gd >= 0.0) {
      double gg = 0.0;
      for (int t = 0; t < N; ++t)
        gg += g[t] * g[t];
      double sc = 1e-3 / std::max(std::sqrt(gg), 1e-300);
      gd = 0.0;
      for (int t = 0; t < N; ++t) {
        d[t] = -g[t] * sc;
        gd += g[t] * d[t];
      }
      k = 0;
    }
    double step = 1.0, En = E;
    bool ok = false;
    for (int ls = 0; ls < 32; ++ls) {
      for (int t = 0; t < N; ++t)
        vn[t] = v[t] + step * d[t];
      En = eval_energy_grad(vn, W, H, 1.0, gn);
      if (En <= E + 1e-4 * step * gd) {
        ok = true;
        break;
      }
      step *= 0.5;
    }
    if (!ok) {
      stall++;
      k = 0;
      if (stall > 3)
        return E;
      continue;
    }
    stall = 0;
    double sy = 0.0;
    for (int t = 0; t < N; ++t) {
      s_[t] = vn[t] - v[t];
      y_[t] = gn[t] - g[t];
      sy += s_[t] * y_[t];
    }
    if (sy > 1e-300) {
      int j = k % m;
      sk[j] = s_;
      yk[j] = y_;
      rho[j] = 1.0 / sy;
      k++;
    }
    double relE = (E - En) / std::max(E, 1e-300);
    v = vn;
    g = gn;
    E = En;
    if (relE < 1e-11 && it > 40)
      return E;
  }
  return E;
}

static bool exact_check_rect(const std::vector<double> &v, double W, double H,
                             double &min_gap, double &min_wall) {
  int n = (int)v.size() / 3;
  min_gap = 1e9;
  min_wall = 1e9;
  std::vector<double> c(n), s(n);
  for (int i = 0; i < n; ++i) {
    c[i] = std::cos(v[2 * n + i]);
    s[i] = std::sin(v[2 * n + i]);
    double h = 0.5 * (std::abs(c[i]) + std::abs(s[i]));
    min_wall = std::min(
        {min_wall, v[i] - h, W - v[i] - h, v[n + i] - h, H - v[n + i] - h});
  }
  if (min_wall < -1e-11)
    return false;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double dx = v[j] - v[i], dy = v[n + j] - v[n + i];
      if (dx * dx + dy * dy >= 2.0)
        continue;
      double best = -1e9;
      for (int k = 0; k < 4; ++k) {
        double ax = (k == 0)   ? c[i]
                    : (k == 1) ? -s[i]
                    : (k == 2) ? c[j]
                               : -s[j];
        double ay = (k == 0) ? s[i] : (k == 1) ? c[i] : (k == 2) ? s[j] : c[j];
        double ri = 0.5 * (std::abs(ax * c[i] + ay * s[i]) +
                           std::abs(-ax * s[i] + ay * c[i]));
        double rj = 0.5 * (std::abs(ax * c[j] + ay * s[j]) +
                           std::abs(-ax * s[j] + ay * c[j]));
        best = std::max(best, std::abs(ax * dx + ay * dy) - (ri + rj));
      }
      min_gap = std::min(min_gap, best);
    }
  }
  return min_gap >= -1e-11 && min_wall >= -1e-11;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: rect_quench IN_TXT OUT_TXT W_FIXED [H_INIT]\n";
    return 1;
  }
  std::ifstream in(argv[1]);
  int n;
  double H0;
  if (!(in >> n >> H0))
    return 1;
  double W = std::stod(argv[3]);
  if (argc >= 5)
    H0 = std::stod(argv[4]);

  std::vector<double> v(3 * n);
  for (int i = 0; i < n; ++i) {
    double x, y, deg;
    in >> x >> y >> deg;
    v[i] = x;
    v[n + i] = y;
    v[2 * n + i] = deg * (M_PI / 180.0);
  }

  double best_H = H0;
  std::vector<double> best_v = v;
  // Try shrinking H inside fixed [0, W]
  double E0 = lbfgs_minimize(best_v, W, best_H, 1200, 1e-26);
  double mg, mw;
  if (E0 < 1e-20 && exact_check_rect(best_v, W, best_H, mg, mw)) {
    // Compress H aggressively
    double cur_H = best_H;
    std::vector<double> cur_v = best_v;
    for (double step_frac :
         {0.02, 0.008, 0.003, 0.001, 0.0004, 0.00015, 0.00005}) {
      for (int rep = 0; rep < 6; ++rep) {
        double try_H = cur_H * (1.0 - step_frac);
        std::vector<double> tv = cur_v;
        double scale_y = try_H / cur_H;
        for (int i = 0; i < n; ++i)
          tv[n + i] *= scale_y;
        double E = lbfgs_minimize(tv, W, try_H, 1000, 1e-26);
        if (E < 1e-22 && exact_check_rect(tv, W, try_H, mg, mw)) {
          cur_H = try_H;
          cur_v = tv;
        } else {
          break;
        }
      }
    }
    best_H = cur_H;
    best_v = cur_v;
  }

  // Recompute exact tight H
  double xmax = -1e9, xmin = 1e9, ymax = -1e9, ymin = 1e9;
  for (int i = 0; i < n; ++i) {
    double h = 0.5 * (std::abs(std::cos(best_v[2 * n + i])) +
                      std::abs(std::sin(best_v[2 * n + i])));
    xmin = std::min(xmin, best_v[i] - h);
    xmax = std::max(xmax, best_v[i] + h);
    ymin = std::min(ymin, best_v[n + i] - h);
    ymax = std::max(ymax, best_v[n + i] + h);
  }
  double W_actual = xmax - xmin;
  double H_actual = ymax - ymin;
  std::ofstream out(argv[2]);
  out << std::setprecision(17) << n << " " << H_actual << "\n";
  for (int i = 0; i < n; ++i) {
    double deg = std::fmod(best_v[2 * n + i] * (180.0 / M_PI), 90.0);
    if (deg < 0)
      deg += 90.0;
    out << (best_v[i] - xmin) << " " << (best_v[n + i] - ymin) << " " << deg
        << "\n";
  }
  std::cout << std::setprecision(12) << "{\"n\":" << n << ",\"W\":" << W_actual
            << ",\"H\":" << H_actual << "}\n";
  return 0;
}
