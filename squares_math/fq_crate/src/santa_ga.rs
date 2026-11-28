//! santa_ga v2: High-throughput compiled Rust port of Jeroen Cottaar's 1st-Place Santa 2025
//! Island Genetic Algorithm adapted for Squares in Squares:
//! - Trig-cached `pen_fast` (zero `sin_cos()` inside the O(pairs) inner loop -> ~4x faster L-BFGS)
//! - Preallocated ring-buffer `LbfgsBuf` (`m = 6`, zero heap allocations per step, `ecut` at `it == 26`)
//! - Exact `mirror_seed` + `mirror_symmetrize` + post-relaxation `mirror_average` (`sym_post`) + `released_mirror`
//! - GLS-Weighted Legalization (`Relaxer2W`: 5 rounds x 220 steps doubling weights on stubborn overlaps)
//! - Multi-step `arm_extend` (`1..3` upright squares appended to a tilted row tip) + `tess_start` resets
//! - Empirical RTX-5090 move weights + Chebyshev block `crossover` + `OVERLAP_TARGETING = 0.80`

use std::env;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_1_SQRT_2, PI, SQRT_2};
use std::fs;
use std::thread;
use std::time::Instant;

const H: f64 = 0.5;

#[inline]
fn sgn(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[derive(Clone)]
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        let mut r = Rng(
            seed.wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(0x632BE59BD9B4E019)
                | 1,
        );
        for _ in 0..8 {
            r.next_u64();
        }
        r
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    #[inline]
    fn f(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
    #[inline]
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
    #[inline]
    fn idx(&mut self, n: usize) -> usize {
        if n <= 1 {
            0
        } else {
            (self.next_u64() as usize) % n
        }
    }
    #[inline]
    fn int_range(&mut self, lo: usize, hi_excl: usize) -> usize {
        if hi_excl <= lo + 1 {
            lo
        } else {
            lo + self.idx(hi_excl - lo)
        }
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.f().max(1e-300);
        let u2 = self.f();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
    fn choice_weighted(&mut self, cdf: &[f64]) -> usize {
        let r = self.f() * cdf.last().copied().unwrap_or(1.0);
        match cdf.binary_search_by(|v| v.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Less)) {
            Ok(i) => i.min(cdf.len().saturating_sub(1)),
            Err(i) => i.min(cdf.len().saturating_sub(1)),
        }
    }
}

#[inline]
fn pen_fast(
    dx: f64,
    dy: f64,
    si: f64,
    ci: f64,
    sj: f64,
    cj: f64,
) -> Option<(f64, [f64; 4])> {
    let mut best = f64::INFINITY;
    let mut bg = [0.0; 4];
    for owner in 0..2 {
        let (so, co, sot, cot) = if owner == 0 {
            (si, ci, sj, cj)
        } else {
            (sj, cj, si, ci)
        };
        for k in 0..2 {
            let (sp, cp) = if k == 0 { (so, co) } else { (co, -so) };
            let proj = dx * cp + dy * sp;
            let ca = cp * cot + sp * sot;
            let sa = sp * cot - cp * sot;
            let o = H + H * (ca.abs() + sa.abs()) - proj.abs();
            if o <= 0.0 {
                return None;
            }
            if o < best {
                best = o;
                let ga = H * (-sgn(ca) * sa + sgn(sa) * ca);
                let sp_ = sgn(proj);
                let dphi = ga - sp_ * (-dx * sp + dy * cp);
                let dother = -ga;
                let ddx = -sp_ * cp;
                let ddy = -sp_ * sp;
                if owner == 0 {
                    bg = [ddx, ddy, dphi, dother];
                } else {
                    bg = [ddx, ddy, dother, dphi];
                }
            }
        }
    }
    Some((best, bg))
}

struct Sys {
    n: usize,
    s: f64,
    skin: f64,
    pairs: Vec<(u32, u32)>,
    refpos: Vec<f64>,
    sc: Vec<(f64, f64)>,
}

impl Sys {
    fn new(n: usize, s: f64) -> Self {
        Sys {
            n,
            s,
            skin: 0.38,
            pairs: Vec::with_capacity(6 * n),
            refpos: vec![f64::NAN; 2 * n],
            sc: vec![(0.0, 1.0); n],
        }
    }

    fn rebuild(&mut self, v: &[f64]) {
        let n = self.n;
        let cut = SQRT_2 + self.skin;
        let cut2 = cut * cut;
        let lo = -1.0;
        let span = (self.s + 2.0).max(2.0);
        let nc = ((span / cut).floor() as usize).max(1);
        let cs = span / nc as f64;
        let cell = |x: f64| -> usize { (((x - lo) / cs).floor().max(0.0) as usize).min(nc - 1) };
        let mut heads = vec![u32::MAX; nc * nc];
        let mut next = vec![u32::MAX; n];
        for i in 0..n {
            let c = cell(v[3 * i]) * nc + cell(v[3 * i + 1]);
            next[i] = heads[c];
            heads[c] = i as u32;
        }
        self.pairs.clear();
        for i in 0..n {
            let (cx, cy) = (cell(v[3 * i]), cell(v[3 * i + 1]));
            for ax in cx.saturating_sub(1)..=(cx + 1).min(nc - 1) {
                for ay in cy.saturating_sub(1)..=(cy + 1).min(nc - 1) {
                    let mut j = heads[ax * nc + ay];
                    while j != u32::MAX {
                        let ju = j as usize;
                        if ju > i {
                            let dx = v[3 * ju] - v[3 * i];
                            let dy = v[3 * ju + 1] - v[3 * i + 1];
                            if dx * dx + dy * dy < cut2 {
                                self.pairs.push((i as u32, j));
                            }
                        }
                        j = next[ju];
                    }
                }
            }
        }
        for i in 0..n {
            self.refpos[2 * i] = v[3 * i];
            self.refpos[2 * i + 1] = v[3 * i + 1];
        }
    }

    #[inline]
    fn check_list(&mut self, v: &[f64]) {
        let lim = (self.skin * 0.5) * (self.skin * 0.5);
        for i in 0..self.n {
            let dx = v[3 * i] - self.refpos[2 * i];
            let dy = v[3 * i + 1] - self.refpos[2 * i + 1];
            if !(dx * dx + dy * dy <= lim) {
                self.rebuild(v);
                return;
            }
        }
    }

    fn energy(&mut self, v: &[f64], g: &mut [f64]) -> f64 {
        self.check_list(v);
        g.fill(0.0);
        let s = self.s;
        let mut e = 0.0;
        for i in 0..self.n {
            let (x, y, t) = (v[3 * i], v[3 * i + 1], v[3 * i + 2]);
            let (st, ct) = t.sin_cos();
            self.sc[i] = (st, ct);
            let w = H * (ct.abs() + st.abs());
            let dw = H * (-sgn(ct) * st + sgn(st) * ct);
            let a = w - x;
            if a > 0.0 {
                e += 4.0 * a * a;
                g[3 * i] -= 8.0 * a;
                g[3 * i + 2] += 8.0 * a * dw;
            }
            let a = x + w - s;
            if a > 0.0 {
                e += 4.0 * a * a;
                g[3 * i] += 8.0 * a;
                g[3 * i + 2] += 8.0 * a * dw;
            }
            let a = w - y;
            if a > 0.0 {
                e += 4.0 * a * a;
                g[3 * i + 1] -= 8.0 * a;
                g[3 * i + 2] += 8.0 * a * dw;
            }
            let a = y + w - s;
            if a > 0.0 {
                e += 4.0 * a * a;
                g[3 * i + 1] += 8.0 * a;
                g[3 * i + 2] += 8.0 * a * dw;
            }
        }
        for &(iu, ju) in &self.pairs {
            let (i, j) = (iu as usize, ju as usize);
            let dx = v[3 * j] - v[3 * i];
            let dy = v[3 * j + 1] - v[3 * i + 1];
            if dx * dx + dy * dy >= 2.0 {
                continue;
            }
            let (si, ci) = self.sc[i];
            let (sj, cj) = self.sc[j];
            if let Some((p, d)) = pen_fast(dx, dy, si, ci, sj, cj) {
                e += p * p;
                let c = 2.0 * p;
                g[3 * j] += c * d[0];
                g[3 * i] -= c * d[0];
                g[3 * j + 1] += c * d[1];
                g[3 * i + 1] -= c * d[1];
                g[3 * i + 2] += c * d[2];
                g[3 * j + 2] += c * d[3];
            }
        }
        e
    }

    fn energy_weighted(
        &mut self,
        v: &[f64],
        w_sq: &[f64],
        g: &mut [f64],
        pen_sq: &mut [f64],
    ) -> f64 {
        self.check_list(v);
        g.fill(0.0);
        pen_sq.fill(0.0);
        let s = self.s;
        let mut e = 0.0;
        for i in 0..self.n {
            let (x, y, t) = (v[3 * i], v[3 * i + 1], v[3 * i + 2]);
            let wi = w_sq[i];
            let (st, ct) = t.sin_cos();
            self.sc[i] = (st, ct);
            let w = H * (ct.abs() + st.abs());
            let dw = H * (-sgn(ct) * st + sgn(st) * ct);
            let a = w - x;
            if a > 0.0 {
                pen_sq[i] = pen_sq[i].max(a);
                e += 4.0 * wi * a * a;
                g[3 * i] -= 8.0 * wi * a;
                g[3 * i + 2] += 8.0 * wi * a * dw;
            }
            let a = x + w - s;
            if a > 0.0 {
                pen_sq[i] = pen_sq[i].max(a);
                e += 4.0 * wi * a * a;
                g[3 * i] += 8.0 * wi * a;
                g[3 * i + 2] += 8.0 * wi * a * dw;
            }
            let a = w - y;
            if a > 0.0 {
                pen_sq[i] = pen_sq[i].max(a);
                e += 4.0 * wi * a * a;
                g[3 * i + 1] -= 8.0 * wi * a;
                g[3 * i + 2] += 8.0 * wi * a * dw;
            }
            let a = y + w - s;
            if a > 0.0 {
                pen_sq[i] = pen_sq[i].max(a);
                e += 4.0 * wi * a * a;
                g[3 * i + 1] += 8.0 * wi * a;
                g[3 * i + 2] += 8.0 * wi * a * dw;
            }
        }
        for &(iu, ju) in &self.pairs {
            let (i, j) = (iu as usize, ju as usize);
            let dx = v[3 * j] - v[3 * i];
            let dy = v[3 * j + 1] - v[3 * i + 1];
            if dx * dx + dy * dy >= 2.0 {
                continue;
            }
            let (si, ci) = self.sc[i];
            let (sj, cj) = self.sc[j];
            if let Some((p, d)) = pen_fast(dx, dy, si, ci, sj, cj) {
                pen_sq[i] = pen_sq[i].max(p);
                pen_sq[j] = pen_sq[j].max(p);
                let wij = 0.5 * (w_sq[i] + w_sq[j]);
                e += wij * p * p;
                let c = 2.0 * wij * p;
                g[3 * j] += c * d[0];
                g[3 * i] -= c * d[0];
                g[3 * j + 1] += c * d[1];
                g[3 * i + 1] -= c * d[1];
                g[3 * i + 2] += c * d[2];
                g[3 * j + 2] += c * d[3];
            }
        }
        e
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

struct LbfgsBuf {
    m: usize,
    g: Vec<f64>,
    d: Vec<f64>,
    xn: Vec<f64>,
    gn: Vec<f64>,
    ss: Vec<Vec<f64>>,
    ys: Vec<Vec<f64>>,
    rho: Vec<f64>,
    alpha: Vec<f64>,
    head: usize,
    len: usize,
}

impl LbfgsBuf {
    fn new(dim: usize, m: usize) -> Self {
        LbfgsBuf {
            m,
            g: vec![0.0; dim],
            d: vec![0.0; dim],
            xn: vec![0.0; dim],
            gn: vec![0.0; dim],
            ss: (0..m).map(|_| vec![0.0; dim]).collect(),
            ys: (0..m).map(|_| vec![0.0; dim]).collect(),
            rho: vec![0.0; m],
            alpha: vec![0.0; m],
            head: 0,
            len: 0,
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    #[inline]
    fn push_pair(&mut self, v: &[f64]) {
        let dim = v.len();
        let mut sy = 0.0;
        for q in 0..dim {
            let sq = self.xn[q] - v[q];
            let yq = self.gn[q] - self.g[q];
            sy += sq * yq;
        }
        if sy > 1e-30 {
            let slot = if self.len < self.m {
                let sl = (self.head + self.len) % self.m;
                self.len += 1;
                sl
            } else {
                let sl = self.head;
                self.head = (self.head + 1) % self.m;
                sl
            };
            for q in 0..dim {
                self.ss[slot][q] = self.xn[q] - v[q];
                self.ys[slot][q] = self.gn[q] - self.g[q];
            }
            self.rho[slot] = 1.0 / sy;
        }
    }
}

fn lbfgs_relax_buf(
    sys: &mut Sys,
    buf: &mut LbfgsBuf,
    v: &mut [f64],
    maxit: usize,
    stop: f64,
    stall_n: usize,
    ecut: Option<f64>,
    w_sq: Option<&[f64]>,
    mut pen_out: Option<&mut [f64]>,
) -> f64 {
    let dim = v.len();
    buf.clear();
    let mut dummy_pen = vec![0.0; if w_sq.is_some() && pen_out.is_none() { sys.n } else { 0 }];
    let mut e = if let Some(ws) = w_sq {
        match pen_out {
            Some(ref mut po) => sys.energy_weighted(v, ws, &mut buf.g, &mut **po),
            None => sys.energy_weighted(v, ws, &mut buf.g, &mut dummy_pen),
        }
    } else {
        sys.energy(v, &mut buf.g)
    };
    let mut fails = 0usize;

    for it in 0..maxit {
        if e <= stop {
            break;
        }
        if it == 26 {
            if let Some(ec) = ecut {
                if e > ec {
                    break;
                }
            }
        }
        for k in 0..dim {
            buf.d[k] = -buf.g[k];
        }
        let h = buf.len;
        for k_ord in (0..h).rev() {
            let sl = (buf.head + k_ord) % buf.m;
            let a = buf.rho[sl] * dot(&buf.ss[sl], &buf.d);
            buf.alpha[k_ord] = a;
            for q in 0..dim {
                buf.d[q] -= a * buf.ys[sl][q];
            }
        }
        if h > 0 {
            let newest = (buf.head + h - 1) % buf.m;
            let yy = dot(&buf.ys[newest], &buf.ys[newest]);
            let gam = if yy > 1e-30 {
                dot(&buf.ss[newest], &buf.ys[newest]) / yy
            } else {
                1.0
            };
            for q in 0..dim {
                buf.d[q] *= gam;
            }
        } else {
            let gmax = buf.g.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let sc = if gmax > 0.0 { (0.05 / gmax).min(1.0) } else { 1.0 };
            for q in 0..dim {
                buf.d[q] *= sc;
            }
        }
        for k_ord in 0..h {
            let sl = (buf.head + k_ord) % buf.m;
            let b = buf.rho[sl] * dot(&buf.ys[sl], &buf.d);
            let diff = buf.alpha[k_ord] - b;
            for q in 0..dim {
                buf.d[q] += buf.ss[sl][q] * diff;
            }
        }
        let mut gd = dot(&buf.g, &buf.d);
        if !(gd < 0.0) {
            buf.clear();
            let gmax = buf.g.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            if gmax == 0.0 {
                break;
            }
            let sc = (0.05 / gmax).min(1.0);
            for q in 0..dim {
                buf.d[q] = -buf.g[q] * sc;
            }
            gd = dot(&buf.g, &buf.d);
        }
        let dmax = buf.d.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let mut step = if dmax > 0.25 { 0.25 / dmax } else { 1.0 };
        let mut ok = false;
        for _ in 0..24 {
            for q in 0..dim {
                buf.xn[q] = v[q] + step * buf.d[q];
            }
            let en = if let Some(ws) = w_sq {
                match pen_out {
                    Some(ref mut po) => sys.energy_weighted(&buf.xn, ws, &mut buf.gn, &mut **po),
                    None => sys.energy_weighted(&buf.xn, ws, &mut buf.gn, &mut dummy_pen),
                }
            } else {
                sys.energy(&buf.xn, &mut buf.gn)
            };
            if en <= e + 1e-4 * step * gd {
                buf.push_pair(v);
                v.copy_from_slice(&buf.xn);
                buf.g.copy_from_slice(&buf.gn);
                let rel = (e - en) / e.abs().max(1e-300);
                e = en;
                ok = true;
                if rel < 1e-9 {
                    fails += 1;
                } else {
                    fails = 0;
                }
                break;
            }
            step *= 0.5;
        }
        if !ok {
            if buf.len == 0 {
                break;
            }
            buf.clear();
            fails += 1;
        }
        if fails > stall_n {
            break;
        }
    }
    e
}

fn exact_side_and_max_pen(v: &[f64]) -> (f64, Vec<f64>, f64) {
    let n = v.len() / 3;
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    let mut sc = Vec::with_capacity(n);
    for i in 0..n {
        let (st, ct) = v[3 * i + 2].sin_cos();
        sc.push((st, ct));
        let h = H * (ct.abs() + st.abs());
        xmin = xmin.min(v[3 * i] - h);
        xmax = xmax.max(v[3 * i] + h);
        ymin = ymin.min(v[3 * i + 1] - h);
        ymax = ymax.max(v[3 * i + 1] + h);
    }
    let mut vn = v.to_vec();
    for i in 0..n {
        vn[3 * i] -= xmin;
        vn[3 * i + 1] -= ymin;
        vn[3 * i + 2] = vn[3 * i + 2].rem_euclid(FRAC_PI_2);
        sc[i] = vn[3 * i + 2].sin_cos();
    }
    let s_exact = (xmax - xmin).max(ymax - ymin);
    let mut sys = Sys::new(n, s_exact + 1.0);
    sys.rebuild(&vn);
    let mut maxp = 0.0f64;
    for &(iu, ju) in &sys.pairs {
        let (i, j) = (iu as usize, ju as usize);
        let dx = vn[3 * j] - vn[3 * i];
        let dy = vn[3 * j + 1] - vn[3 * i + 1];
        if dx * dx + dy * dy >= 2.0 {
            continue;
        }
        let (si, ci) = sc[i];
        let (sj, cj) = sc[j];
        if let Some((p, _)) = pen_fast(dx, dy, si, ci, sj, cj) {
            maxp = maxp.max(p);
        }
    }
    (s_exact, vn, maxp)
}

fn fit_box(v: &[f64], l: f64) -> Vec<f64> {
    let (s, mut q, _) = exact_side_and_max_pen(v);
    let n = q.len() / 3;
    let sc = (l - 1.0) / (s - 1.0).max(1e-9);
    for i in 0..n {
        q[3 * i] = 0.5 + (q[3 * i] - 0.5) * sc;
        q[3 * i + 1] = 0.5 + (q[3 * i + 1] - 0.5) * sc;
    }
    q
}

fn rescale_vec(v: &[f64], l_old: f64, l_new: f64) -> Vec<f64> {
    let n = v.len() / 3;
    let mut y = v.to_vec();
    let sc = (l_new - 1.0) / (l_old - 1.0).max(1e-9);
    for i in 0..n {
        y[3 * i] = 0.5 + (y[3 * i] - 0.5) * sc;
        y[3 * i + 1] = 0.5 + (y[3 * i + 1] - 0.5) * sc;
    }
    y
}

fn strip_angle(m: usize, w: f64) -> Option<f64> {
    let mf = (m + 1) as f64;
    let mut lo = (1.0 / mf).atan();
    let mut hi = FRAC_PI_4;
    let g = |t: f64| mf * t.cos() + t.sin() - w;
    if g(lo) < 0.0 || g(hi) > 0.0 {
        return None;
    }
    for _ in 0..55 {
        let mid = 0.5 * (lo + hi);
        if g(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

fn tess_or_window_start(n: usize, l: f64, p_best: &[f64], rng: &mut Rng) -> Vec<f64> {
    let k_int = l.floor() as isize;
    let f = l - (k_int as f64);
    let u = rng.f();
    if u < 0.35 {
        // Window local melt on p_best
        let mut q = fit_box(p_best, l);
        let w = rng.uniform(3.0, ((k_int as f64) * 0.45).max(3.5));
        let x0 = rng.uniform(0.0, (l - w).max(0.1));
        let y0 = rng.uniform(0.0, (l - w).max(0.1));
        for i in 0..n {
            if q[3 * i] > x0 && q[3 * i] < x0 + w && q[3 * i + 1] > y0 && q[3 * i + 1] < y0 + w {
                q[3 * i] = rng.uniform(x0 + 0.5, (x0 + w - 0.5).max(x0 + 0.55));
                q[3 * i + 1] = rng.uniform(y0 + 0.5, (y0 + w - 0.5).max(y0 + 0.55));
                q[3 * i + 2] = rng.uniform(0.0, FRAC_PI_2);
            }
        }
        return q;
    }
    let mut pts: Vec<(f64, f64, f64)> = Vec::with_capacity(n + 32);
    if u < 0.55 {
        // diag45 band
        let s = FRAC_1_SQRT_2;
        let w = rng.int_range(2, 6) as f64;
        let off = rng.uniform(0.0, 2.0 * s);
        for i in (-2 * k_int - 4)..(2 * k_int + 4) {
            for j in -6..7 {
                let x = off + ((i - j) as f64) * s;
                let y = off + ((i + j) as f64) * s;
                if x >= s && x <= l - s && y >= s && y <= l - s && (x - y).abs() * s <= 0.5 * w - 0.5 + 1e-9 {
                    pts.push((x, y, FRAC_PI_4));
                }
            }
        }
        for i in 0..=k_int {
            for j in 0..=k_int {
                let x = (i as f64) + 0.5;
                let y = (j as f64) + 0.5;
                if x <= l - 0.5 && y <= l - 0.5 && (x - y).abs() * s >= 0.5 * w + s - 1e-9 {
                    pts.push((x, y, 0.0));
                }
            }
        }
    } else {
        // block or band strip-law start
        let is_band = u < 0.75;
        let (m, w_strip) = if is_band {
            (k_int.max(2) as usize, l)
        } else {
            let m_c = rng.int_range(2, (k_int.max(4) as usize) - 1);
            (m_c, (m_c as f64) + f)
        };
        let t = if is_band || rng.f() < 0.75 {
            strip_angle(m, w_strip).unwrap_or(FRAC_PI_4)
        } else {
            FRAC_PI_4
        };
        let (s_, c_) = t.sin_cos();
        let h = 0.5 * (c_ + s_);
        let per = (((w_strip - s_) / c_ + 1e-9).floor().max(0.0)) as usize;
        let hb = if is_band {
            rng.uniform(0.25, 0.58) * l
        } else {
            rng.uniform(0.40, 0.88) * l
        };
        let y0 = if is_band { rng.uniform(0.0, (l - hb).max(0.0)) } else { 0.0 };
        let mut r = 0usize;
        while per > 0 {
            let yb = y0 + h + (r as f64) / c_;
            if yb + ((per - 1) as f64) * s_ + h > y0 + hb + 1e-9 {
                break;
            }
            for k in 0..per {
                pts.push((h + (k as f64) * c_, yb + (k as f64) * s_, t));
            }
            r += 1;
        }
        let ytop = if r > 0 {
            y0 + ((r - 1) as f64) / c_ + ((per - 1) as f64) * s_ + 2.0 * h
        } else {
            y0
        };
        for i in 0..(k_int.max(1) as usize) {
            for j in 0..=(k_int.max(1) as usize) {
                let x = l - 0.5 - (i as f64);
                let y = l - 0.5 - (j as f64);
                if x < 0.5 - 1e-9 || y < 0.5 - 1e-9 {
                    continue;
                }
                let inside = (y + 0.5 > y0 && y - 0.5 < ytop) && (is_band || x - 0.5 < w_strip);
                if !inside {
                    pts.push((x, y, 0.0));
                }
            }
        }
    }
    if pts.len() > n {
        let drop = pts.len() - n;
        for d_i in 0..drop {
            let last_p = pts.len() - 1 - d_i;
            let rm = rng.idx(last_p + 1);
            pts.swap(rm, last_p);
        }
        pts.truncate(n);
    } else {
        while pts.len() < n {
            pts.push((
                rng.uniform(0.5, l - 0.5),
                rng.uniform(0.5, l - 0.5),
                rng.uniform(0.0, FRAC_PI_2),
            ));
        }
    }
    let flip_x = rng.f() < 0.5;
    let flip_y = rng.f() < 0.5;
    let swap_xy = rng.f() < 0.5;
    let mut out = Vec::with_capacity(3 * n);
    for (mut x, mut y, mut a) in pts {
        if flip_x {
            x = l - x;
            a = (-a).rem_euclid(FRAC_PI_2);
        }
        if flip_y {
            y = l - y;
            a = (-a).rem_euclid(FRAC_PI_2);
        }
        if swap_xy {
            std::mem::swap(&mut x, &mut y);
            a = (-a).rem_euclid(FRAC_PI_2);
        }
        out.extend_from_slice(&[x.clamp(0.5, l - 0.5), y.clamp(0.5, l - 0.5), a]);
    }
    out
}

fn mirror_seed(p_in: &[f64], l: f64, rng: &mut Rng) -> (Vec<f64>, usize) {
    let n = p_in.len() / 3;
    let mut q = fit_box(p_in, l);
    if rng.f() < 0.5 {
        for i in 0..n {
            q[3 * i] = l - q[3 * i];
            q[3 * i + 2] = (-q[3 * i + 2]).rem_euclid(FRAC_PI_2);
        }
    }
    let mut dist: Vec<(usize, f64)> = (0..n)
        .map(|i| (i, (q[3 * i + 1] - q[3 * i]) / SQRT_2))
        .collect();
    let mut r = dist.iter().filter(|&&(_, d)| d.abs() < 0.35).count();
    if (n - r) % 2 == 1 {
        if r < n {
            r += 1;
        } else {
            r -= 1;
        }
    }
    dist.sort_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap());
    let diag_idx: Vec<usize> = dist[..r].iter().map(|&(i, _)| i).collect();
    let mut rest = dist[r..].to_vec();
    rest.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let h = (n - r) / 2;
    let mut out = Vec::with_capacity(3 * n);
    for &(i, _) in &rest[..h] {
        out.extend_from_slice(&[q[3 * i], q[3 * i + 1], q[3 * i + 2].rem_euclid(FRAC_PI_2)]);
    }
    for k in 0..h {
        let (gx, gy, ga) = (out[3 * k], out[3 * k + 1], out[3 * k + 2]);
        out.extend_from_slice(&[gy, gx, (-ga).rem_euclid(FRAC_PI_2)]);
    }
    for &i in &diag_idx {
        let m = 0.5 * (q[3 * i] + q[3 * i + 1]);
        let am = q[3 * i + 2].rem_euclid(FRAC_PI_2);
        let da = if am > FRAC_PI_4 * 0.5 && am < FRAC_PI_4 * 1.5 {
            FRAC_PI_4
        } else {
            0.0
        };
        out.extend_from_slice(&[m, m, da]);
    }
    (out, r)
}

fn mirror_symmetrize(v: &mut [f64], v0: &[f64], n: usize, r: usize) {
    let h = (n - r) / 2;
    for k in 0..h {
        let p = h + k;
        let dg = (v[3 * k] - v0[3 * k]).abs()
            + (v[3 * k + 1] - v0[3 * k + 1]).abs()
            + (v[3 * k + 2] - v0[3 * k + 2]).abs();
        let dp = (v[3 * p] - v0[3 * p]).abs()
            + (v[3 * p + 1] - v0[3 * p + 1]).abs()
            + (v[3 * p + 2] - v0[3 * p + 2]).abs();
        if dp > dg {
            let (px, py, pa) = (v[3 * p], v[3 * p + 1], v[3 * p + 2]);
            v[3 * k] = py;
            v[3 * k + 1] = px;
            v[3 * k + 2] = (-pa).rem_euclid(FRAC_PI_2);
        } else {
            let (gx, gy, ga) = (v[3 * k], v[3 * k + 1], v[3 * k + 2]);
            v[3 * p] = gy;
            v[3 * p + 1] = gx;
            v[3 * p + 2] = (-ga).rem_euclid(FRAC_PI_2);
        }
    }
    for i in (2 * h)..n {
        let m = 0.5 * (v[3 * i] + v[3 * i + 1]);
        let am = v[3 * i + 2].rem_euclid(FRAC_PI_2);
        v[3 * i] = m;
        v[3 * i + 1] = m;
        v[3 * i + 2] = if am > FRAC_PI_4 * 0.5 && am < FRAC_PI_4 * 1.5 {
            FRAC_PI_4
        } else {
            0.0
        };
    }
}

fn mirror_average(v: &mut [f64], n: usize, r: usize) {
    let h = (n - r) / 2;
    for k in 0..h {
        let p = h + k;
        let gx = 0.5 * (v[3 * k] + v[3 * p + 1]);
        let gy = 0.5 * (v[3 * k + 1] + v[3 * p]);
        let a1 = v[3 * k + 2].rem_euclid(FRAC_PI_2);
        let a2 = (-v[3 * p + 2]).rem_euclid(FRAC_PI_2);
        let diff = (a2 - a1 + FRAC_PI_4).rem_euclid(FRAC_PI_2) - FRAC_PI_4;
        let ga = (a1 + 0.5 * diff).rem_euclid(FRAC_PI_2);
        v[3 * k] = gx;
        v[3 * k + 1] = gy;
        v[3 * k + 2] = ga;
        v[3 * p] = gy;
        v[3 * p + 1] = gx;
        v[3 * p + 2] = (-ga).rem_euclid(FRAC_PI_2);
    }
    for i in (2 * h)..n {
        let m = 0.5 * (v[3 * i] + v[3 * i + 1]);
        let am = v[3 * i + 2].rem_euclid(FRAC_PI_2);
        v[3 * i] = m;
        v[3 * i + 1] = m;
        v[3 * i + 2] = if am > FRAC_PI_4 * 0.5 && am < FRAC_PI_4 * 1.5 {
            FRAC_PI_4
        } else {
            0.0
        };
    }
}

fn descriptor(v: &[f64], l: f64) -> [f32; 128] {
    let n = v.len() / 3;
    let g = 8usize;
    let mut cnt = [0.0f32; 64];
    let mut ang = [0.0f32; 64];
    for i in 0..n {
        let ix = (((v[3 * i] / l) * g as f64).floor().clamp(0.0, 7.0)) as usize;
        let iy = (((v[3 * i + 1] / l) * g as f64).floor().clamp(0.0, 7.0)) as usize;
        let c = ix * g + iy;
        cnt[c] += 1.0;
        let a = ((v[3 * i + 2] + FRAC_PI_4).rem_euclid(FRAC_PI_2) - FRAC_PI_4).abs() as f32;
        ang[c] += a;
    }
    let mut out = [0.0f32; 128];
    let norm = (n as f32 / 64.0).max(1e-9);
    for c in 0..64 {
        out[c] = cnt[c] / norm;
        out[64 + c] = if cnt[c] > 0.0 {
            4.0 * ang[c] / cnt[c]
        } else {
            0.0
        };
    }
    out
}

fn desc_l1(a: &[f32; 128], b: &[f32; 128]) -> f32 {
    let mut s = 0.0f32;
    for k in 0..128 {
        s += (a[k] - b[k]).abs();
    }
    s
}

fn residual_cdf(v: &[f64], l: f64) -> Option<Vec<f64>> {
    let n = v.len() / 3;
    let mut sys = Sys::new(n, l);
    sys.rebuild(v);
    let mut p_sq = vec![0.0f64; n];
    let mut sc = Vec::with_capacity(n);
    for i in 0..n {
        let (x, y, t) = (v[3 * i], v[3 * i + 1], v[3 * i + 2]);
        let (st, ct) = t.sin_cos();
        sc.push((st, ct));
        let w = H * (ct.abs() + st.abs());
        let wv = (w - x).max(x + w - l).max(w - y).max(y + w - l).max(0.0);
        p_sq[i] = wv;
    }
    for &(iu, ju) in &sys.pairs {
        let (i, j) = (iu as usize, ju as usize);
        let dx = v[3 * j] - v[3 * i];
        let dy = v[3 * j + 1] - v[3 * i + 1];
        if dx * dx + dy * dy >= 2.0 {
            continue;
        }
        let (si, ci) = sc[i];
        let (sj, cj) = sc[j];
        if let Some((p, _)) = pen_fast(dx, dy, si, ci, sj, cj) {
            p_sq[i] = p_sq[i].max(p);
            p_sq[j] = p_sq[j].max(p);
        }
    }
    let mx = p_sq.iter().cloned().fold(0.0f64, f64::max);
    if mx <= 1e-7 {
        return None;
    }
    let mut spread = p_sq.clone();
    for &(iu, ju) in &sys.pairs {
        let (i, j) = (iu as usize, ju as usize);
        let dx = v[3 * j] - v[3 * i];
        let dy = v[3 * j + 1] - v[3 * i + 1];
        if dx * dx + dy * dy < 2.56 {
            spread[i] = spread[i].max(p_sq[j]);
            spread[j] = spread[j].max(p_sq[i]);
        }
    }
    let mut cdf = Vec::with_capacity(n);
    let mut acc = 0.0;
    for &val in &spread {
        acc += val;
        cdf.push(acc);
    }
    if acc <= 0.0 {
        None
    } else {
        Some(cdf)
    }
}

#[inline]
fn angle_dist_90(a: f64, b: f64) -> f64 {
    ((a - b + FRAC_PI_4).rem_euclid(FRAC_PI_2) - FRAC_PI_4).abs()
}

fn mutate_one(
    v: &mut [f64],
    l: f64,
    kind: usize,
    res_cdf: Option<&[f64]>,
    rng: &mut Rng,
) {
    let n = v.len() / 3;
    let pick_sq = |rng: &mut Rng| -> usize {
        if let Some(cdf) = res_cdf {
            if rng.f() < 0.80 {
                return rng.choice_weighted(cdf);
            }
        }
        rng.idx(n)
    };
    let pick_centre = |v: &[f64], rng: &mut Rng| -> (f64, f64) {
        if res_cdf.is_some() {
            let s = pick_sq(rng);
            (
                v[3 * s] + rng.uniform(-0.5, 0.5),
                v[3 * s + 1] + rng.uniform(-0.5, 0.5),
            )
        } else {
            (rng.uniform(0.0, l), rng.uniform(0.0, l))
        }
    };

    match kind {
        1 | 2 => {
            let (xy, th) = if kind == 1 {
                (0.05, PI / 6.0)
            } else {
                (0.10, PI)
            };
            let i = pick_sq(rng);
            v[3 * i] += rng.uniform(-xy, xy);
            v[3 * i + 1] += rng.uniform(-xy, xy);
            v[3 * i + 2] += rng.uniform(-th, th);
        }
        3 | 4 => {
            let (xy, th) = if kind == 3 {
                (0.05, PI / 6.0)
            } else {
                (0.10, PI)
            };
            let (cx, cy) = pick_centre(v, rng);
            let m = rng.int_range(2, 9).min(n);
            let mut d2: Vec<(usize, f64)> = (0..n)
                .map(|i| {
                    let (dx, dy) = (v[3 * i] - cx, v[3 * i + 1] - cy);
                    (i, dx * dx + dy * dy)
                })
                .collect();
            d2.select_nth_unstable_by(m - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
            for &(i, _) in &d2[..m] {
                v[3 * i] += rng.uniform(-xy, xy);
                v[3 * i + 1] += rng.uniform(-xy, xy);
                v[3 * i + 2] += rng.uniform(-th, th);
            }
        }
        6 => {
            let (cx, cy) = pick_centre(v, rng);
            let rad = rng.uniform(0.5, 2.0);
            let amp = rng.uniform(-PI, PI);
            for i in 0..n {
                let (dx, dy) = (v[3 * i] - cx, v[3 * i + 1] - cy);
                let dist = dx.hypot(dy);
                if dist < rad {
                    let t = amp * (1.0 - dist / rad);
                    let (st, ct) = t.sin_cos();
                    v[3 * i] = cx + ct * dx - st * dy;
                    v[3 * i + 1] = cy + st * dx + ct * dy;
                    v[3 * i + 2] += t;
                }
            }
        }
        8 => {
            // row_slide (#1 empirical move)
            let i0 = rng.idx(n);
            let (x0, y0, a0) = (v[3 * i0], v[3 * i0 + 1], v[3 * i0 + 2]);
            let ang = a0 + (rng.idx(2) as f64) * FRAC_PI_2;
            let (uy, ux) = ang.sin_cos();
            let dl = rng.uniform(-0.30, 0.30);
            for i in 0..n {
                if angle_dist_90(v[3 * i + 2], a0) < 0.01 {
                    let perp = -(v[3 * i] - x0) * uy + (v[3 * i + 1] - y0) * ux;
                    if perp.abs() < 0.30 {
                        v[3 * i] += dl * ux;
                        v[3 * i + 1] += dl * uy;
                    }
                }
            }
        }
        9 => {
            // band_reangle (#2 empirical move)
            let tilted: Vec<usize> = (0..n)
                .filter(|&i| angle_dist_90(v[3 * i + 2], 0.0) > 0.01)
                .collect();
            if !tilted.is_empty() {
                let i0 = tilted[rng.idx(tilted.len())];
                let a0 = v[3 * i0 + 2];
                let sel: Vec<usize> = (0..n)
                    .filter(|&i| angle_dist_90(v[3 * i + 2], a0) < 0.01)
                    .collect();
                let cnt = sel.len() as f64;
                let mx = sel.iter().map(|&i| v[3 * i]).sum::<f64>() / cnt;
                let my = sel.iter().map(|&i| v[3 * i + 1]).sum::<f64>() / cnt;
                let th = rng.uniform(-0.021, 0.021);
                let (st, ct) = th.sin_cos();
                for i in sel {
                    let (dx, dy) = (v[3 * i] - mx, v[3 * i + 1] - my);
                    v[3 * i] = mx + ct * dx - st * dy;
                    v[3 * i + 1] = my + st * dx + ct * dy;
                    v[3 * i + 2] += th;
                }
            }
        }
        11 => {
            let (cx, cy) = (rng.uniform(0.0, l), rng.uniform(0.0, l));
            let hw = rng.uniform(1.5, (l * 0.25).max(2.0));
            let hh = rng.uniform(1.5, (l * 0.25).max(2.0));
            let flip_x = rng.idx(2) == 0;
            for i in 0..n {
                if (v[3 * i] - cx).abs() < hw && (v[3 * i + 1] - cy).abs() < hh {
                    if flip_x {
                        v[3 * i] = 2.0 * cx - v[3 * i];
                    } else {
                        v[3 * i + 1] = 2.0 * cy - v[3 * i + 1];
                    }
                    v[3 * i + 2] = -v[3 * i + 2];
                }
            }
        }
        12 => {
            let w = rng.uniform(2.0, (l / 3.0).max(2.5));
            let h = rng.uniform(2.0, (l / 3.0).max(2.5));
            let (ax, ay) = (rng.f() * (l - w).max(0.1), rng.f() * (l - h).max(0.1));
            let (bx, by) = (rng.f() * (l - w).max(0.1), rng.f() * (l - h).max(0.1));
            for i in 0..n {
                let in_a = v[3 * i] >= ax
                    && v[3 * i] < ax + w
                    && v[3 * i + 1] >= ay
                    && v[3 * i + 1] < ay + h;
                let in_b = v[3 * i] >= bx
                    && v[3 * i] < bx + w
                    && v[3 * i + 1] >= by
                    && v[3 * i + 1] < by + h;
                if in_a && !in_b {
                    v[3 * i] += bx - ax;
                    v[3 * i + 1] += by - ay;
                } else if in_b && !in_a {
                    v[3 * i] += ax - bx;
                    v[3 * i + 1] += ay - by;
                }
            }
        }
        13 => {
            let (cx, cy) = pick_centre(v, rng);
            let m = rng.int_range(1, 10).min(n);
            let mut d2: Vec<(usize, f64)> = (0..n)
                .map(|i| {
                    let (dx, dy) = (v[3 * i] - cx, v[3 * i + 1] - cy);
                    (i, dx * dx + dy * dy)
                })
                .collect();
            d2.select_nth_unstable_by(m - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
            let cnt = m as f64;
            let mx = d2[..m].iter().map(|&(i, _)| v[3 * i]).sum::<f64>() / cnt;
            let my = d2[..m].iter().map(|&(i, _)| v[3 * i + 1]).sum::<f64>() / cnt;
            let th = if rng.f() < 0.5 { FRAC_PI_4 } else { -FRAC_PI_4 };
            let (st, ct) = th.sin_cos();
            for &(i, _) in &d2[..m] {
                let (dx, dy) = (v[3 * i] - mx, v[3 * i + 1] - my);
                v[3 * i] = mx + ct * dx - st * dy;
                v[3 * i + 1] = my + st * dx + ct * dy;
                v[3 * i + 2] += th;
            }
        }
        14 => {
            if n >= 4 {
                let i0 = pick_sq(rng);
                let (x0, y0, th) = (v[3 * i0], v[3 * i0 + 1], v[3 * i0 + 2]);
                let mut d2: Vec<(usize, f64)> = (0..n)
                    .map(|i| {
                        if i == i0 {
                            (i, -1.0)
                        } else {
                            let (dx, dy) = (v[3 * i] - x0, v[3 * i + 1] - y0);
                            (i, dx * dx + dy * dy)
                        }
                    })
                    .collect();
                d2.select_nth_unstable_by(3, |a, b| a.1.partial_cmp(&b.1).unwrap());
                let mut sel: Vec<usize> = d2[..4].iter().map(|&(i, _)| i).collect();
                let mx = sel.iter().map(|&i| v[3 * i]).sum::<f64>() / 4.0;
                let my = sel.iter().map(|&i| v[3 * i + 1]).sum::<f64>() / 4.0;
                sel.sort_by(|&a, &b| {
                    let aa = ((v[3 * a + 1] - my).atan2(v[3 * a] - mx) - th + PI)
                        .rem_euclid(2.0 * PI)
                        - PI;
                    let bb = ((v[3 * b + 1] - my).atan2(v[3 * b] - mx) - th + PI)
                        .rem_euclid(2.0 * PI)
                        - PI;
                    aa.partial_cmp(&bb).unwrap()
                });
                let (st, ct) = th.sin_cos();
                let offs = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)];
                for (j, &si) in sel.iter().enumerate() {
                    let (ox, oy) = offs[j];
                    v[3 * si] = mx + ct * ox - st * oy;
                    v[3 * si + 1] = my + st * ox + ct * oy;
                    v[3 * si + 2] = th;
                }
            }
        }
        15 => {
            let i0 = pick_sq(rng);
            let (x0, y0, a0) = (v[3 * i0], v[3 * i0 + 1], v[3 * i0 + 2]);
            let mut same: Vec<(usize, f64)> = (0..n)
                .filter(|&i| angle_dist_90(v[3 * i + 2], a0) < 0.01)
                .map(|i| {
                    let (dx, dy) = (v[3 * i] - x0, v[3 * i + 1] - y0);
                    (i, dx * dx + dy * dy)
                })
                .collect();
            let m = rng.int_range(4, 41).min(same.len());
            if m >= 2 {
                same.select_nth_unstable_by(m - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
                let cnt = m as f64;
                let mx = same[..m].iter().map(|&(i, _)| v[3 * i]).sum::<f64>() / cnt;
                let my = same[..m].iter().map(|&(i, _)| v[3 * i + 1]).sum::<f64>() / cnt;
                let deg = if rng.f() < 0.3 {
                    rng.uniform(5.0, 20.0)
                } else {
                    rng.uniform(0.5, 3.0)
                } * if rng.f() < 0.5 { -1.0 } else { 1.0 };
                let th = deg.to_radians();
                let (st, ct) = th.sin_cos();
                for &(i, _) in &same[..m] {
                    let (dx, dy) = (v[3 * i] - mx, v[3 * i + 1] - my);
                    v[3 * i] = mx + ct * dx - st * dy;
                    v[3 * i + 1] = my + st * dx + ct * dy;
                    v[3 * i + 2] += th;
                }
            }
        }
        16 => {
            // Multi-step arm_extend (1..4 upright squares appended onto a tilted row tip + neighbor nudge!)
            let tilted: Vec<usize> = (0..n)
                .filter(|&i| angle_dist_90(v[3 * i + 2], 0.0) > 0.01)
                .collect();
            if !tilted.is_empty() {
                let i0 = tilted[rng.idx(tilted.len())];
                let (x0, y0, a0) = (v[3 * i0], v[3 * i0 + 1], v[3 * i0 + 2]);
                let dr = a0 + (rng.idx(4) as f64) * FRAC_PI_2;
                let (uy, ux) = dr.sin_cos();
                let mut end_along = -1e9f64;
                for i in 0..n {
                    if angle_dist_90(v[3 * i + 2], a0) < 0.01 {
                        let (rx, ry) = (v[3 * i] - x0, v[3 * i + 1] - y0);
                        let along = rx * ux + ry * uy;
                        let perp = -rx * uy + ry * ux;
                        if perp.abs() < 0.30 && along > end_along {
                            end_along = along;
                        }
                    }
                }
                if end_along > -1e8 {
                    let n_ext = if rng.f() < 0.45 { 1 } else { rng.int_range(2, 5) };
                    let mut converted = Vec::with_capacity(n_ext);
                    for step_k in 1..=n_ext {
                        let tx = x0 + (end_along + step_k as f64) * ux;
                        let ty = y0 + (end_along + step_k as f64) * uy;
                        if tx < 0.45 || tx > l - 0.45 || ty < 0.45 || ty > l - 0.45 {
                            break;
                        }
                        let mut best_j = usize::MAX;
                        let mut best_d2 = 2.25f64;
                        for j in 0..n {
                            if angle_dist_90(v[3 * j + 2], 0.0) <= 0.01 && !converted.contains(&j) {
                                let d2 =
                                    (v[3 * j] - tx).powi(2) + (v[3 * j + 1] - ty).powi(2);
                                if d2 < best_d2 {
                                    best_d2 = d2;
                                    best_j = j;
                                }
                            }
                        }
                        if best_j != usize::MAX {
                            v[3 * best_j] = tx;
                            v[3 * best_j + 1] = ty;
                            v[3 * best_j + 2] = a0;
                            converted.push(best_j);
                        } else {
                            break;
                        }
                    }
                    // Gently nudge nearby upright neighbors outward so L-BFGS clears corner saddles cleanly
                    for &cj in &converted {
                        let (cx, cy) = (v[3 * cj], v[3 * cj + 1]);
                        for k in 0..n {
                            if k != cj && angle_dist_90(v[3 * k + 2], 0.0) <= 0.01 && !converted.contains(&k) {
                                let dx = v[3 * k] - cx;
                                let dy = v[3 * k + 1] - cy;
                                let d2 = dx * dx + dy * dy;
                                if d2 > 1e-6 && d2 < 1.38 {
                                    let inv = 0.085 / d2.sqrt();
                                    v[3 * k] += dx * inv;
                                    v[3 * k + 1] += dy * inv;
                                }
                            }
                        }
                    }
                }
            }
        }
        17 | _ => {
            // parallel_arm_spawn: nucleate a parallel tilted row (2..5 squares) adjacent to an existing tilted row!
            let tilted: Vec<usize> = (0..n)
                .filter(|&i| angle_dist_90(v[3 * i + 2], 0.0) > 0.01)
                .collect();
            if !tilted.is_empty() {
                let i0 = tilted[rng.idx(tilted.len())];
                let (x0, y0, a0) = (v[3 * i0], v[3 * i0 + 1], v[3 * i0 + 2]);
                let dr = a0 + (rng.idx(2) as f64) * FRAC_PI_2;
                let (uy, ux) = dr.sin_cos();
                let side_sgn = if rng.f() < 0.5 { 1.0 } else { -1.0 };
                let (nx_p, ny_p) = (-side_sgn * uy, side_sgn * ux);
                let bx = x0 + nx_p;
                let by = y0 + ny_p;
                let len_row = rng.int_range(2, 6);
                let mut used = Vec::with_capacity(len_row);
                let start_off = -0.5 * ((len_row - 1) as f64);
                for sk in 0..len_row {
                    let s_along = start_off + (sk as f64);
                    let tx = bx + s_along * ux;
                    let ty = by + s_along * uy;
                    if tx < 0.45 || tx > l - 0.45 || ty < 0.45 || ty > l - 0.45 {
                        continue;
                    }
                    let mut best_j = usize::MAX;
                    let mut best_d2 = 1.96f64;
                    for j in 0..n {
                        if angle_dist_90(v[3 * j + 2], 0.0) <= 0.01 && !used.contains(&j) {
                            let d2 = (v[3 * j] - tx).powi(2) + (v[3 * j + 1] - ty).powi(2);
                            if d2 < best_d2 {
                                best_d2 = d2;
                                best_j = j;
                            }
                        }
                    }
                    if best_j != usize::MAX {
                        v[3 * best_j] = tx;
                        v[3 * best_j + 1] = ty;
                        v[3 * best_j + 2] = a0;
                        used.push(best_j);
                    }
                }
            } else {
                // Pure grid fallback: tilt a small collinear strip of 3..5 squares at strip-law angle
                let i0 = rng.idx(n);
                let (x0, y0) = (v[3 * i0], v[3 * i0 + 1]);
                let a_new = (1.0f64 / (rng.int_range(3, 8) as f64)).atan();
                let (uy, ux) = a_new.sin_cos();
                let mut d2: Vec<(usize, f64)> = (0..n)
                    .map(|i| {
                        let (rx, ry) = (v[3 * i] - x0, v[3 * i + 1] - y0);
                        let perp = (-rx * uy + ry * ux).abs();
                        let along = (rx * ux + ry * uy).abs();
                        (i, perp * 4.0 + along)
                    })
                    .collect();
                let m = rng.int_range(3, 6).min(n);
                d2.select_nth_unstable_by(m - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
                for &(i, _) in &d2[..m] {
                    v[3 * i + 2] = a_new;
                }
            }
        }
    }
}

fn crossover_one(child: &mut [f64], mate: &[f64], l: f64, rng: &mut Rng) {
    let n = child.len() / 3;
    let k_max = ((0.30 * n as f64) as usize).max(2);
    let k = rng.int_range(2, k_max + 1).min(n);
    let s = l * ((k as f64) / (n as f64)).sqrt();
    let ccx = rng.f() * (l - s).max(0.0) + 0.5 * s;
    let ccy = rng.f() * (l - s).max(0.0) + 0.5 * s;
    let mcx = rng.f() * (l - s).max(0.0) + 0.5 * s;
    let mcy = rng.f() * (l - s).max(0.0) + 0.5 * s;

    let mut oc: Vec<(usize, f64)> = (0..n)
        .map(|i| (i, (child[3 * i] - ccx).abs().max((child[3 * i + 1] - ccy).abs())))
        .collect();
    let mut om: Vec<(usize, f64)> = (0..n)
        .map(|i| (i, (mate[3 * i] - mcx).abs().max((mate[3 * i + 1] - mcy).abs())))
        .collect();
    oc.select_nth_unstable_by(k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
    om.select_nth_unstable_by(k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());

    let kf = k as f64;
    let cen_cx = oc[..k].iter().map(|&(i, _)| child[3 * i]).sum::<f64>() / kf;
    let cen_cy = oc[..k].iter().map(|&(i, _)| child[3 * i + 1]).sum::<f64>() / kf;
    let cen_mx = om[..k].iter().map(|&(i, _)| mate[3 * i]).sum::<f64>() / kf;
    let cen_my = om[..k].iter().map(|&(i, _)| mate[3 * i + 1]).sum::<f64>() / kf;

    let mir = rng.f() < 0.5;
    let rot = (rng.idx(4) as f64) * FRAC_PI_2;
    let (sr, cr) = rot.sin_cos();

    for q in 0..k {
        let ic = oc[q].0;
        let im = om[q].0;
        let dx = mate[3 * im] - cen_mx;
        let mut dy = mate[3 * im + 1] - cen_my;
        let mut th = mate[3 * im + 2];
        if mir {
            dy = -dy;
            th = -th;
        }
        child[3 * ic] = cen_cx + cr * dx - sr * dy;
        child[3 * ic + 1] = cen_cy + sr * dx + cr * dy;
        child[3 * ic + 2] = th + rot;
    }
}

fn read_cfg(path: &str) -> (usize, f64, Vec<f64>) {
    let txt = fs::read_to_string(path).expect("read --in");
    let mut lines = txt.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let hd: Vec<f64> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|w| w.parse().unwrap())
        .collect();
    let (n, s) = (hd[0] as usize, hd[1]);
    let mut v = Vec::with_capacity(3 * n);
    for l in lines.take(n) {
        let row: Vec<&str> = l.split_whitespace().collect();
        let x: f64 = row[0].parse().unwrap();
        let y: f64 = row[1].parse().unwrap();
        let deg: f64 = row[2].parse().unwrap();
        v.extend_from_slice(&[x, y, deg.to_radians()]);
    }
    assert_eq!(v.len(), 3 * n);
    (n, s, v)
}

fn write_cfg_atomic(path: &str, s: f64, v: &[f64]) {
    let n = v.len() / 3;
    let mut out = format!("{} {:.17}\n", n, s);
    for i in 0..n {
        let deg = v[3 * i + 2].to_degrees().rem_euclid(90.0);
        out += &format!("{:.17} {:.17} {:.15}\n", v[3 * i], v[3 * i + 1], deg);
    }
    let tmp = format!("{path}.tmp");
    if fs::write(&tmp, out).is_ok() {
        let _ = fs::rename(&tmp, path);
    }
}

fn arg<T: std::str::FromStr>(a: &[String], k: &str, def: T) -> T {
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(def)
}

fn multi_args(a: &[String], k: &str) -> Vec<String> {
    let mut out = Vec::new();
    for i in 0..a.len() {
        if a[i] == k {
            if let Some(v) = a.get(i + 1) {
                out.push(v.clone());
            }
        }
    }
    out
}

#[derive(Clone)]
struct Island {
    l: f64,
    pop: Vec<Vec<f64>>,
    ene: Vec<f64>,
    e0: f64,
    e_ref: f64,
    last_imp: usize,
    born: usize,
    sym: usize, // 0 = free, 2 + r = diagonal mirror with r diagonal squares
    released: bool,
    shr: usize,
    hold_until: usize,
    role: u8,
}

fn exact_bbox_side(v: &[f64]) -> f64 {
    let n = v.len() / 3;
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for i in 0..n {
        let (st, ct) = v[3 * i + 2].sin_cos();
        let h = H * (ct.abs() + st.abs());
        xmin = xmin.min(v[3 * i] - h);
        xmax = xmax.max(v[3 * i] + h);
        ymin = ymin.min(v[3 * i + 1] - h);
        ymax = ymax.max(v[3 * i + 1] + h);
    }
    (xmax - xmin).max(ymax - ymin)
}

fn tilted_sum(v: &[f64]) -> f64 {
    let n = v.len() / 3;
    let mut s = 0.0;
    for i in 0..n {
        s += ((v[3 * i + 2] + FRAC_PI_4).rem_euclid(FRAC_PI_2) - FRAC_PI_4).abs();
    }
    s
}

// Tuple: (coords, box_l, ecut_opt, sym_code, out_energy)
fn relax_batch_parallel(
    batch: &mut [(Vec<f64>, f64, Option<f64>, usize, f64)],
    maxit: usize,
    stop: f64,
    nthreads: usize,
) {
    let len = batch.len();
    if len == 0 {
        return;
    }
    let dim = batch[0].0.len();
    let nt = nthreads.max(1).min(len);
    let chunk_sz = (len + nt - 1) / nt;
    thread::scope(|s| {
        for chunk in batch.chunks_mut(chunk_sz) {
            s.spawn(move || {
                let mut buf = LbfgsBuf::new(dim, 6);
                let mut g_tmp = vec![0.0; dim];
                for item in chunk.iter_mut() {
                    let n = item.0.len() / 3;
                    let mut sys = Sys::new(n, item.1);
                    let mut e = lbfgs_relax_buf(
                        &mut sys,
                        &mut buf,
                        &mut item.0,
                        maxit,
                        stop,
                        18,
                        item.2,
                        None,
                        None,
                    );
                    if item.3 >= 2 {
                        mirror_average(&mut item.0, n, item.3 - 2);
                        e = sys.energy(&item.0, &mut g_tmp);
                    }
                    item.4 = e;
                }
            });
        }
    });
}

fn build_fine_legalize_batch(isl: &[Island], best_legal_s: f64) -> Vec<(Vec<f64>, f64, f64)> {
    let rel_rungs = [0.0, 2e-5, 7e-5, 1.8e-4, 4e-4, 9e-4, 2.2e-3];
    let sub_best = [4.5e-4, 2.2e-4, 9e-5, 3e-5, 8e-6, 2e-6];
    let mut leg_batch: Vec<(Vec<f64>, f64, f64)> = Vec::new();
    for d in isl {
        let mut cands: Vec<f64> = Vec::with_capacity(14);
        for &dl in &rel_rungs {
            let lk = d.l + dl;
            if lk < best_legal_s - 1e-9 {
                cands.push(lk);
            }
        }
        for &db in &sub_best {
            let lk = best_legal_s - db;
            if lk >= d.l - 1e-12 && lk < best_legal_s - 1e-9 {
                cands.push(lk);
            }
        }
        cands.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut prev = -1e9f64;
        for lk in cands {
            if lk - prev > 1.5e-6 {
                leg_batch.push((rescale_vec(&d.pop[0], d.l, lk), lk, f64::INFINITY));
                prev = lk;
            }
        }
    }
    leg_batch
}

fn gls_legalize_batch(
    batch: &mut [(Vec<f64>, f64, f64)],
    nthreads: usize,
) {
    let len = batch.len();
    if len == 0 {
        return;
    }
    let dim = batch[0].0.len();
    let nt = nthreads.max(1).min(len);
    let chunk_sz = (len + nt - 1) / nt;
    thread::scope(|s| {
        for chunk in batch.chunks_mut(chunk_sz) {
            s.spawn(move || {
                let mut buf = LbfgsBuf::new(dim, 6);
                for item in chunk.iter_mut() {
                    let n = item.0.len() / 3;
                    let mut sys = Sys::new(n, item.1);
                    let mut e = lbfgs_relax_buf(
                        &mut sys,
                        &mut buf,
                        &mut item.0,
                        950,
                        1e-26,
                        24,
                        None,
                        None,
                        None,
                    );
                    if e > 1e-22 {
                        let (_, _, mp0) = exact_side_and_max_pen(&item.0);
                        if mp0 < 5e-3 {
                            // Relaxer2W: 5 rounds x 220 steps doubling weights on stubborn overlaps!
                            let mut w_sq = vec![1.0f64; n];
                            let mut pen_sq = vec![0.0f64; n];
                            for _rnd in 0..5 {
                                lbfgs_relax_buf(
                                    &mut sys,
                                    &mut buf,
                                    &mut item.0,
                                    220,
                                    1e-26,
                                    18,
                                    None,
                                    Some(&w_sq),
                                    Some(&mut pen_sq),
                                );
                                let max_p = pen_sq.iter().cloned().fold(0.0f64, f64::max);
                                if max_p <= 1e-12 {
                                    break;
                                }
                                for i in 0..n {
                                    if pen_sq[i] > 1.5e-6 {
                                        w_sq[i] = (w_sq[i] * 2.0).min(1e3);
                                    } else {
                                        w_sq[i] = (w_sq[i] * 0.9).max(1.0);
                                    }
                                }
                            }
                            e = lbfgs_relax_buf(
                                &mut sys,
                                &mut buf,
                                &mut item.0,
                                450,
                                1e-26,
                                22,
                                None,
                                None,
                                None,
                            );
                        }
                    }
                    item.2 = e;
                }
            });
        }
    });
}

fn role_for_island(idx: usize, n_islands: usize) -> u8 {
    if n_islands <= 1 {
        return 0;
    }
    if n_islands == 2 {
        return if idx == 0 { 0 } else { 1 };
    }
    match idx % 6 {
        0 => 0, // Tight compression
        1 => 1, // Medium breathing arm-growth (+0.0042, 28-gen hold)
        2 => 2, // Wide breathing / window-melt arm-growth (+0.0095, 32-gen hold)
        3 => 0, // Alt-seed tight compression
        4 => 3, // Breathing diagonal mirror (+0.0036, 20-gen hold)
        _ => 4, // Tight diagonal mirror
    }
}

fn init_island(
    seeds: &[(f64, Vec<f64>)],
    seed_idx: usize,
    role: u8,
    best_legal_s: f64,
    use_tess: bool,
    pop_sz: usize,
    gen: usize,
    steps: usize,
    thr: f64,
    nthreads: usize,
    rng: &mut Rng,
) -> Island {
    let sym_flag = role == 3 || role == 4;
    let (s_best0, ref p_best0) = seeds[0];
    let n = p_best0.len() / 3;
    if use_tess && !sym_flag {
        let mut lt = best_legal_s.min(s_best0) + 0.015;
        let base_t = tess_or_window_start(n, lt, p_best0, rng);
        let mut batch: Vec<(Vec<f64>, f64, Option<f64>, usize, f64)> = (0..pop_sz)
            .map(|m| {
                let mut x = base_t.clone();
                if m > 0 {
                    for k in 0..3 * n {
                        x[k] += 0.008 * rng.gauss();
                    }
                }
                (x, lt, None, 0, f64::INFINITY)
            })
            .collect();
        relax_batch_parallel(&mut batch, steps, 1e-26, nthreads);
        batch.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));
        while batch[0].4 > thr && lt < best_legal_s + 0.044 {
            let l_next = lt + 0.006;
            for it in batch.iter_mut() {
                it.0 = rescale_vec(&it.0, lt, l_next);
                it.1 = l_next;
                it.4 = f64::INFINITY;
            }
            lt = l_next;
            relax_batch_parallel(&mut batch, steps, 1e-26, nthreads);
            batch.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));
        }
        if batch[0].4 <= thr * 5.0 {
            let e0 = batch[0].4;
            return Island {
                l: lt,
                pop: batch.iter().map(|it| it.0.clone()).collect(),
                ene: batch.iter().map(|it| it.4).collect(),
                e0,
                e_ref: e0,
                last_imp: gen,
                born: gen,
                sym: 0,
                released: false,
                shr: 0,
                hold_until: gen + 24,
                role,
            };
        }
    }

    let (s_seed, ref p0) = seeds[seed_idx % seeds.len()];
    let (l0, hold_gens, init_noise) = match role {
        1 => (best_legal_s + 0.0042, 28usize, 0.0040),
        2 => (best_legal_s + 0.0095, 32usize, 0.0060),
        3 => (best_legal_s + 0.0036, 20usize, 0.0035),
        4 => (s_seed.min(best_legal_s + 0.0010), 0usize, 0.0025),
        _ => {
            let tight_l = if seed_idx == 0 {
                best_legal_s - 5.0e-5
            } else {
                s_seed.min(best_legal_s + 0.0015)
            };
            (tight_l, 0usize, 0.0025)
        }
    };
    let (base, sym_code) = if sym_flag && n >= 4 {
        let (b, r) = mirror_seed(p0, l0, rng);
        (b, 2 + r)
    } else {
        (fit_box(p0, l0), 0)
    };

    let mut batch: Vec<(Vec<f64>, f64, Option<f64>, usize, f64)> = Vec::with_capacity(pop_sz);
    for m in 0..pop_sz {
        let mut x = base.clone();
        if m > 0 {
            for k in 0..3 * n {
                x[k] += init_noise * rng.gauss();
            }
            if sym_code >= 2 {
                mirror_symmetrize(&mut x, &base, n, sym_code - 2);
            }
            for sq_i in 0..n {
                x[3 * sq_i] = x[3 * sq_i].clamp(0.5, l0 - 0.5);
                x[3 * sq_i + 1] = x[3 * sq_i + 1].clamp(0.5, l0 - 0.5);
            }
        }
        batch.push((x, l0, None, sym_code, f64::INFINITY));
    }
    relax_batch_parallel(&mut batch, steps, 1e-26, nthreads);
    batch.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));
    let e0 = batch[0].4;
    let pop = batch.iter().map(|it| it.0.clone()).collect();
    let ene = batch.iter().map(|it| it.4).collect();
    Island {
        l: l0,
        pop,
        ene,
        e0,
        e_ref: e0,
        last_imp: gen,
        born: gen,
        sym: sym_code,
        released: false,
        shr: 0,
        hold_until: gen + hold_gens,
        role,
    }
}

fn select_diverse(
    xa: Vec<Vec<f64>>,
    ea: Vec<f64>,
    l: f64,
    pop_sz: usize,
    elite: usize,
    div_thr: f32,
    warmup: bool,
    seed_tilt: f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let tot = xa.len();
    let sort_keys: Vec<(u8, f64)> = (0..tot)
        .map(|i| {
            let e = ea[i];
            if e <= 1e-18 {
                let sb = exact_bbox_side(&xa[i]);
                if warmup {
                    let nov = (tilted_sum(&xa[i]) - seed_tilt).abs().min(8.0);
                    (0u8, sb - 0.00045 * nov)
                } else {
                    (0u8, sb)
                }
            } else {
                (1u8, e)
            }
        })
        .collect();
    let cmp_idx = |a: usize, b: usize| -> std::cmp::Ordering {
        sort_keys[a]
            .0
            .cmp(&sort_keys[b].0)
            .then_with(|| sort_keys[a].1.partial_cmp(&sort_keys[b].1).unwrap_or(std::cmp::Ordering::Equal))
    };
    let mut order: Vec<usize> = (0..tot).collect();
    order.sort_by(|&a, &b| cmp_idx(a, b));
    let descs: Vec<[f32; 128]> = xa.iter().map(|v| descriptor(v, l)).collect();

    let el = elite.min(pop_sz).min(tot);
    let mut keep: Vec<usize> = order[..el].to_vec();
    let mut mind = vec![f32::INFINITY; tot];
    for &k in &keep {
        for j in 0..tot {
            mind[j] = mind[j].min(desc_l1(&descs[j], &descs[k]));
        }
    }
    let eff_div = if warmup { div_thr * 0.72 } else { div_thr };
    for &j in &order[el..] {
        if keep.len() >= pop_sz {
            break;
        }
        if mind[j] >= eff_div {
            keep.push(j);
            for m in 0..tot {
                mind[m] = mind[m].min(desc_l1(&descs[m], &descs[j]));
            }
        }
    }
    if keep.len() < pop_sz {
        for &j in &order {
            if keep.len() >= pop_sz {
                break;
            }
            if !keep.contains(&j) {
                keep.push(j);
            }
        }
    }
    keep.sort_by(|&a, &b| cmp_idx(a, b));
    let out_x = keep.iter().map(|&i| xa[i].clone()).collect();
    let out_e = keep.iter().map(|&i| ea[i]).collect();
    (out_x, out_e)
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let inp: String = arg(&a, "--in", String::new());
    let out: String = arg(&a, "--out", String::from("santa_out.txt"));
    let seed: u64 = arg(&a, "--seed", 1);
    let sec_budget: f64 = arg(&a, "--sec", 600.0);
    let n_islands: usize = arg(&a, "--islands", 6);
    let pop_sz: usize = arg(&a, "--pop", 14);
    let off_sz: usize = arg(&a, "--offspring", 28);
    let elite: usize = arg(&a, "--elite", 5);
    let steps: usize = arg(&a, "--steps", 125);
    let soft: f64 = arg(&a, "--soft", 1e-8);
    let shrink: f64 = arg(&a, "--shrink", 7.5e-5);
    let div_thr: f32 = arg(&a, "--div", 0.55);
    let nthreads: usize = arg(&a, "--threads", 4);
    let nbr_files = multi_args(&a, "--nbr");

    let t0 = Instant::now();
    let mut rng = Rng::new(seed);
    let (n, _s_hdr, v_best) = read_cfg(&inp);
    let (mut best_legal_s, v_norm, _) = exact_side_and_max_pen(&v_best);
    let s_start = best_legal_s;
    let seed_tilt0 = tilted_sum(&v_norm);

    let mut seeds: Vec<(f64, Vec<f64>)> = vec![(best_legal_s, v_norm)];
    for nf in &nbr_files {
        if let Ok(txt) = fs::read_to_string(nf) {
            if !txt.trim().is_empty() {
                let (m, _, vm) = read_cfg(nf);
                if m > n {
                    let drop = m - n;
                    for _ in 0..3 {
                        let mut idx: Vec<usize> = (0..m).collect();
                        for d_i in 0..drop {
                            let last_pos = m - 1 - d_i;
                            let rm = rng.idx(last_pos + 1);
                            idx.swap(rm, last_pos);
                        }
                        idx.truncate(n);
                        idx.sort_unstable();
                        let mut vk = Vec::with_capacity(3 * n);
                        for &i in &idx {
                            vk.extend_from_slice(&vm[3 * i..3 * i + 3]);
                        }
                        let (sk, vkn, _) = exact_side_and_max_pen(&vk);
                        seeds.push((sk, vkn));
                    }
                } else if m == n {
                    let (sk, vkn, mp_m) = exact_side_and_max_pen(&vm);
                    let s_eff = if mp_m <= 1e-9 {
                        sk
                    } else {
                        sk.max(best_legal_s) + 1e-5
                    };
                    seeds.push((s_eff, vkn));
                }
            }
        }
    }

    // Tight compression move weights + Breathing arm-growth warmup move weights (boosts 16=arm_extend, 17=parallel_arm_spawn, 15=cluster_tilt)
    let move_kinds: [usize; 13] = [8, 9, 16, 17, 1, 6, 2, 15, 4, 11, 3, 13, 14];
    let move_wts_tight: [f64; 13] = [4.56, 3.98, 3.15, 1.80, 3.07, 2.33, 2.08, 1.25, 0.87, 0.84, 0.78, 0.65, 0.55];
    let move_wts_breath: [f64; 13] = [4.10, 3.40, 5.40, 3.90, 1.20, 2.15, 1.00, 2.75, 1.40, 1.15, 0.80, 1.85, 1.65];
    let mut move_cdf_tight = Vec::with_capacity(13);
    let mut move_cdf_breath = Vec::with_capacity(13);
    let mut acc_t = 0.0;
    let mut acc_b = 0.0;
    for k in 0..13 {
        acc_t += move_wts_tight[k];
        acc_b += move_wts_breath[k];
        move_cdf_tight.push(acc_t);
        move_cdf_breath.push(acc_b);
    }

    let thr = soft * (n as f64);
    let mut isl: Vec<Island> = (0..n_islands)
        .map(|i| {
            let role = role_for_island(i, n_islands);
            let sidx = if i == 0 { 0 } else { i % seeds.len() };
            init_island(
                &seeds,
                sidx,
                role,
                best_legal_s,
                false,
                pop_sz,
                0,
                steps,
                thr,
                nthreads,
                &mut rng,
            )
        })
        .collect();

    let mut gen = 0usize;
    let mut wins = 0usize;
    let mut resets = 0usize;
    let mut rep_swaps = 0usize;
    let mut best_champ_score = f64::INFINITY;

    while t0.elapsed().as_secs_f64() < sec_budget {
        let r_stall = 90 + gen / 10;
        let best_isl = (0..n_islands)
            .min_by(|&a, &b| {
                isl[a]
                    .l
                    .partial_cmp(&isl[b].l)
                    .unwrap()
                    .then_with(|| isl[a].e0.partial_cmp(&isl[b].e0).unwrap())
            })
            .unwrap_or(0);
        let champs: Vec<(Vec<f64>, f64, f64)> = isl
            .iter()
            .map(|d| (d.pop[0].clone(), d.l, d.e0))
            .collect();

        let mut all_children: Vec<(Vec<f64>, f64, Option<f64>, usize, f64)> =
            Vec::with_capacity(n_islands * off_sz);
        for ii in 0..n_islands {
            let l = isl[ii].l;
            let sym = isl[ii].sym;
            let in_warmup = gen < isl[ii].hold_until;
            let res_cdf = residual_cdf(&isl[ii].pop[0], l);
            let wk = isl[ii].ene.last().copied().unwrap_or(f64::INFINITY);
            let ecut = if !in_warmup && wk.is_finite() && wk > 0.0 {
                Some(12.0 * wk)
            } else {
                None
            };
            let nx = if sym > 0 {
                0
            } else {
                ((0.10 * off_sz as f64).round() as usize).min(off_sz)
            };
            let stalled = gen.saturating_sub(isl[ii].last_imp) > r_stall / 2;
            let mcdf = if in_warmup { &move_cdf_breath } else { &move_cdf_tight };

            for c_i in 0..off_sz {
                let p_idx = rng.idx(isl[ii].pop.len());
                let mut child = isl[ii].pop[p_idx].clone();
                let x0 = child.clone();
                if c_i < nx {
                    let use_own = rng.f() < 0.65 || n_islands <= 1;
                    if use_own {
                        let m_idx = rng.idx(isl[ii].pop.len());
                        crossover_one(&mut child, &isl[ii].pop[m_idx], l, &mut rng);
                    } else {
                        let hop = 1 + rng.idx(n_islands - 1);
                        let mi = (ii + hop) % n_islands;
                        if stalled || in_warmup || (champs[mi].1, champs[mi].2) >= (l, isl[ii].e0) {
                            let mate_scaled = rescale_vec(&champs[mi].0, champs[mi].1, l);
                            crossover_one(&mut child, &mate_scaled, l, &mut rng);
                        } else {
                            let m_idx = rng.idx(isl[ii].pop.len());
                            crossover_one(&mut child, &isl[ii].pop[m_idx], l, &mut rng);
                        }
                    }
                } else {
                    let mk = move_kinds[rng.choice_weighted(mcdf)];
                    mutate_one(&mut child, l, mk, res_cdf.as_deref(), &mut rng);
                    if in_warmup && rng.f() < 0.35 {
                        // Two-move topological combo inside breathing warmup (e.g. arm_extend + row_slide)
                        let mk2 = move_kinds[rng.choice_weighted(mcdf)];
                        mutate_one(&mut child, l, mk2, None, &mut rng);
                    }
                }
                if sym >= 2 {
                    mirror_symmetrize(&mut child, &x0, n, sym - 2);
                }
                for sq_i in 0..n {
                    child[3 * sq_i] = child[3 * sq_i].clamp(0.5, l - 0.5);
                    child[3 * sq_i + 1] = child[3 * sq_i + 1].clamp(0.5, l - 0.5);
                }
                all_children.push((child, l, ecut, sym, f64::INFINITY));
            }
        }

        relax_batch_parallel(&mut all_children, steps, 1e-26, nthreads);

        for ii in 0..n_islands {
            let sl = &all_children[ii * off_sz..(ii + 1) * off_sz];
            let l = isl[ii].l;
            let sym = isl[ii].sym;
            let in_warmup = gen < isl[ii].hold_until;
            let mut xa = isl[ii].pop.clone();
            let mut ea = isl[ii].ene.clone();
            for (ch_v, _, _, _, ch_e) in sl {
                if *ch_e <= 1e-22 {
                    let (s_ex, vn, mp) = exact_side_and_max_pen(ch_v);
                    if mp <= 1e-12 && s_ex < best_legal_s - 1e-10 {
                        best_legal_s = s_ex;
                        wins += 1;
                        write_cfg_atomic(&out, best_legal_s, &vn);
                    }
                }
                xa.push(ch_v.clone());
                ea.push(*ch_e);
            }
            let (new_pop, new_ene) =
                select_diverse(xa, ea, l, pop_sz, elite, div_thr, in_warmup, seed_tilt0);
            isl[ii].pop = new_pop;
            isl[ii].ene = new_ene;
            isl[ii].e0 = isl[ii].ene[0];
            if isl[ii].e0 < 0.5 * isl[ii].e_ref || in_warmup {
                isl[ii].e_ref = isl[ii].e0;
                isl[ii].last_imp = gen;
            }
            if !in_warmup {
                let c_score = isl[ii].l + 15.0 * (isl[ii].e0 / (n as f64)).max(0.0).sqrt();
                if c_score < best_champ_score - 1e-7 && isl[ii].e0 < 1e-3 * (n as f64) {
                    best_champ_score = c_score;
                    let (s_ch, v_ch, _) = exact_side_and_max_pen(&isl[ii].pop[0]);
                    write_cfg_atomic(&format!("{out}.champ"), s_ch.min(isl[ii].l), &v_ch);
                }
            }
            if gen >= isl[ii].hold_until && isl[ii].e0 < thr {
                let mut stp = shrink * if isl[ii].e0 < thr * 0.01 { 4.0 } else { 1.0 };
                let excess = l - best_legal_s;
                if excess > 0.0004 {
                    stp = stp.max(0.14 * excess).max(2.2e-4);
                }
                let l_new = l - stp;
                let mut re_batch: Vec<(Vec<f64>, f64, Option<f64>, usize, f64)> = isl[ii]
                    .pop
                    .iter()
                    .map(|p| (rescale_vec(p, l, l_new), l_new, None, sym, f64::INFINITY))
                    .collect();
                relax_batch_parallel(&mut re_batch, steps, 1e-26, nthreads);
                re_batch.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));
                isl[ii].l = l_new;
                isl[ii].pop = re_batch.iter().map(|it| it.0.clone()).collect();
                isl[ii].ene = re_batch.iter().map(|it| it.4).collect();
                isl[ii].e0 = isl[ii].ene[0];
                isl[ii].e_ref = isl[ii].e0;
                isl[ii].last_imp = gen;
                isl[ii].shr += 1;
            }
        }

        // Pressure-Ladder Replica Exchange (Odriozola / Basurto downward injection every 10 gens!)
        if gen % 10 == 9 && n_islands >= 2 {
            let champ_descs: Vec<[f32; 128]> =
                (0..n_islands).map(|i| descriptor(&isl[i].pop[0], isl[i].l)).collect();
            let mut rep_targets: Vec<usize> = Vec::new();
            let mut rep_batch: Vec<(Vec<f64>, f64, Option<f64>, usize, f64)> = Vec::new();
            for hi in 0..n_islands {
                if isl[hi].sym != 0 || isl[hi].e0 > thr * 2.5 {
                    continue;
                }
                for lo in 0..n_islands {
                    if hi == lo || isl[lo].sym != 0 {
                        continue;
                    }
                    let dl = isl[hi].l - isl[lo].l;
                    if dl > 4.0e-5 && dl <= 0.0075 {
                        if desc_l1(&champ_descs[hi], &champ_descs[lo]) >= 0.30 {
                            rep_targets.push(lo);
                            rep_batch.push((
                                rescale_vec(&isl[hi].pop[0], isl[hi].l, isl[lo].l),
                                isl[lo].l,
                                None,
                                0,
                                f64::INFINITY,
                            ));
                        }
                    }
                }
            }
            if !rep_batch.is_empty() {
                relax_batch_parallel(&mut rep_batch, 240, 1e-26, nthreads);
                for (k_rep, &lo) in rep_targets.iter().enumerate() {
                    let (ref rv, rl, _, _, re) = rep_batch[k_rep];
                    if re <= 1e-22 && rl < best_legal_s - 1e-10 {
                        let (s_ex, vn, mp) = exact_side_and_max_pen(rv);
                        if mp <= 1e-12 && s_ex < best_legal_s - 1e-10 {
                            best_legal_s = s_ex;
                            wins += 1;
                            write_cfg_atomic(&out, best_legal_s, &vn);
                        }
                    }
                    let med_e = isl[lo].ene[isl[lo].ene.len() / 2];
                    if re < med_e {
                        let mut xa = isl[lo].pop.clone();
                        let mut ea = isl[lo].ene.clone();
                        xa.push(rv.clone());
                        ea.push(re);
                        let in_w = gen < isl[lo].hold_until;
                        let (npop, nene) =
                            select_diverse(xa, ea, isl[lo].l, pop_sz, elite, div_thr, in_w, seed_tilt0);
                        isl[lo].pop = npop;
                        isl[lo].ene = nene;
                        isl[lo].e0 = isl[lo].ene[0];
                        rep_swaps += 1;
                    }
                }
            }
        }

        // Periodic Fine-Rung GLS-Weighted Legalization (`Relaxer2W`) every 12 generations!
        if gen % 12 == 11 {
            let mut leg_batch = build_fine_legalize_batch(&isl, best_legal_s);
            if !leg_batch.is_empty() {
                gls_legalize_batch(&mut leg_batch, nthreads);
                for (lv, _, le) in &leg_batch {
                    if *le <= 1e-20 {
                        let (s_ex, vn, mp) = exact_side_and_max_pen(lv);
                        if mp <= 1e-12 && s_ex < best_legal_s - 1e-10 {
                            best_legal_s = s_ex;
                            wins += 1;
                            write_cfg_atomic(&out, best_legal_s, &vn);
                        }
                    }
                }
            }
        }

        // Island resets, tess_start topology injection, & Released-Mirror transfer!
        let min_l = isl.iter().map(|d| d.l).fold(f64::INFINITY, f64::min);
        for ii in 0..n_islands {
            if ii == best_isl || gen < isl[ii].hold_until {
                continue;
            }
            let age = gen.saturating_sub(isl[ii].born);
            let stall = gen.saturating_sub(isl[ii].last_imp);
            if stall > r_stall && isl[ii].sym >= 2 && !isl[ii].released && rng.f() < 0.55 {
                if let Some(jj) = (0..n_islands)
                    .filter(|&j| j != ii && j != best_isl && isl[j].sym == 0)
                    .max_by(|&a, &b| isl[a].l.partial_cmp(&isl[b].l).unwrap())
                {
                    let nxy = rng.uniform(0.002, 0.010);
                    let nth = rng.uniform(0.2, 1.0).to_radians();
                    let l_rel = isl[ii].l;
                    let role_jj = isl[jj].role;
                    let mut rel_batch: Vec<(Vec<f64>, f64, Option<f64>, usize, f64)> = isl[ii]
                        .pop
                        .iter()
                        .map(|p| {
                            let mut x = p.clone();
                            for sq_i in 0..n {
                                x[3 * sq_i] += nxy * rng.gauss();
                                x[3 * sq_i + 1] += nxy * rng.gauss();
                                x[3 * sq_i + 2] += nth * rng.gauss();
                            }
                            (x, l_rel, None, 0, f64::INFINITY)
                        })
                        .collect();
                    relax_batch_parallel(&mut rel_batch, steps, 1e-26, nthreads);
                    rel_batch.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));
                    let e0_r = rel_batch[0].4;
                    isl[jj] = Island {
                        l: l_rel,
                        pop: rel_batch.iter().map(|it| it.0.clone()).collect(),
                        ene: rel_batch.iter().map(|it| it.4).collect(),
                        e0: e0_r,
                        e_ref: e0_r,
                        last_imp: gen,
                        born: gen,
                        sym: 0,
                        released: false,
                        shr: 0,
                        hold_until: gen,
                        role: role_jj,
                    };
                    isl[ii].released = true;
                    isl[ii].last_imp = gen;
                    continue;
                }
            }
            let post_hold = gen.saturating_sub(isl[ii].hold_until);
            let hopeless = post_hold >= 140 && isl[ii].e0 > 1000.0 * thr;
            let far_behind = post_hold >= 85 && isl[ii].l - min_l > 0.0020;
            if hopeless || far_behind || stall > r_stall {
                resets += 1;
                let role = isl[ii].role;
                let use_tess = (role == 2 && rng.f() < 0.45) || (role == 1 && rng.f() < 0.20);
                let sidx = rng.idx(seeds.len());
                isl[ii] = init_island(
                    &seeds,
                    sidx,
                    role,
                    best_legal_s,
                    use_tess,
                    pop_sz,
                    gen,
                    steps,
                    thr,
                    nthreads,
                    &mut rng,
                );
            }
        }

        gen += 1;
    }

    // Final end-of-visit Fine-Rung GLS-Weighted Legalization on all island champions!
    let mut final_leg = build_fine_legalize_batch(&isl, best_legal_s);
    if !final_leg.is_empty() {
        gls_legalize_batch(&mut final_leg, nthreads);
        for (lv, _, le) in &final_leg {
            if *le <= 1e-20 {
                let (s_ex, vn, mp) = exact_side_and_max_pen(lv);
                if mp <= 1e-12 && s_ex < best_legal_s - 1e-10 {
                    best_legal_s = s_ex;
                    wins += 1;
                    write_cfg_atomic(&out, best_legal_s, &vn);
                }
            }
        }
    }

    println!(
        "{{\"n\": {}, \"gens\": {}, \"wins\": {}, \"resets\": {}, \"swaps\": {}, \"s_start\": {:.12}, \"s_best\": {:.12}, \"sec\": {:.2}}}",
        n,
        gen,
        wins,
        resets,
        rep_swaps,
        s_start,
        best_legal_s,
        t0.elapsed().as_secs_f64()
    );
}
