package com.nvidia.spark.rapids.fusion

/** Device-side function library compiled into every fused kernel. No headers: NVRTC built-ins only. */
object Prelude {
  val text: String = """
typedef signed char i8; typedef short i16; typedef int i32; typedef long long i64;
typedef unsigned int u32; typedef unsigned long long u64; typedef float f32; typedef double f64;
#define DEV static __device__ __forceinline__

DEV void bail(int* f) { if (f != nullptr) *f = 1; }   // domain guard tripped: host discards the batch and re-runs unfused

// ---- loads/stores (bool is a byte in cuDF) ----
DEV bool ld_bool(const bool* p, int i, bool inb) { return inb ? (((const unsigned char*)p)[i] != 0) : false; }
template<class T> DEV T ld_t(const T* p, int i, bool inb) { return inb ? p[i] : (T)0; }
DEV int ld_i8(const i8* p, int i, bool inb) { return inb ? (int)p[i] : 0; }      // narrow ints are held as int in registers
DEV int ld_i16(const i16* p, int i, bool inb) { return inb ? (int)p[i] : 0; }
DEV i32 ld_i32(const i32* p, int i, bool inb) { return ld_t(p, i, inb); }
DEV i64 ld_i64(const i64* p, int i, bool inb) { return ld_t(p, i, inb); }
DEV f32 ld_f32(const f32* p, int i, bool inb) { return ld_t(p, i, inb); }
DEV f64 ld_f64(const f64* p, int i, bool inb) { return ld_t(p, i, inb); }
DEV i32 ld_date(const i32* p, int i, bool inb) { return ld_t(p, i, inb); }
DEV i64 ld_ts(const i64* p, int i, bool inb) { return ld_t(p, i, inb); }
DEV bool vld(const u32* m, int i) { return m == nullptr || ((m[i >> 5] >> (i & 31)) & 1u) != 0u; }
DEV unsigned char st_bool(bool v) { return v ? 1 : 0; }
DEV i8 st_i8(int v) { return (i8)v; } DEV i16 st_i16(int v) { return (i16)v; } DEV i32 st_i32(i32 v) { return v; } DEV i64 st_i64(i64 v) { return v; }
DEV f32 st_f32(f32 v) { return v; } DEV f64 st_f64(f64 v) { return v; } DEV i32 st_date(i32 v) { return v; } DEV i64 st_ts(i64 v) { return v; }

// sign-extend the low 8/16 bits (inline PTX so the optimizer cannot narrow the following compares into 16-bit PTX ops)
DEV int norm8(int x) { int r; asm("bfe.s32 %0, %1, 0, 8;" : "=r"(r) : "r"(x)); return r; }
DEV int norm16(int x) { int r; asm("bfe.s32 %0, %1, 0, 16;" : "=r"(r) : "r"(x)); return r; }

// ---- wrap-around integer arithmetic, IEEE float arithmetic (no FMA contraction: compiled with --fmad=false) ----
// modular arithmetic: 64-bit only for 64-bit types, 32-bit unsigned math for i8/i16/i32 (narrow types: no 64-bit detour)
template<class T> DEV T w_add(T a, T b) { return sizeof(T) == 8 ? (T)((u64)(i64)a + (u64)(i64)b) : (T)((u32)(i32)a + (u32)(i32)b); }
template<class T> DEV T w_sub(T a, T b) { return sizeof(T) == 8 ? (T)((u64)(i64)a - (u64)(i64)b) : (T)((u32)(i32)a - (u32)(i32)b); }
template<class T> DEV T w_mul(T a, T b) { return sizeof(T) == 8 ? (T)((u64)(i64)a * (u64)(i64)b) : (T)((u32)(i32)a * (u32)(i32)b); }
template<class T> DEV T w_neg(T a) { return sizeof(T) == 8 ? (T)(0ull - (u64)(i64)a) : (T)(0u - (u32)(i32)a); }
template<class T> DEV T w_abs(T a) { return a < 0 ? w_neg(a) : a; }
DEV f32 w_add(f32 a, f32 b) { return __fadd_rn(a, b); } DEV f64 w_add(f64 a, f64 b) { return __dadd_rn(a, b); }
DEV f32 w_sub(f32 a, f32 b) { return __fsub_rn(a, b); } DEV f64 w_sub(f64 a, f64 b) { return __dsub_rn(a, b); }
DEV f32 w_mul(f32 a, f32 b) { return __fmul_rn(a, b); } DEV f64 w_mul(f64 a, f64 b) { return __dmul_rn(a, b); }
DEV f32 w_neg(f32 a) { return __fsub_rn(0.0f, a); } DEV f64 w_neg(f64 a) { return __dsub_rn(0.0, a); }
DEV f32 w_abs(f32 a) { return fabsf(a); } DEV f64 w_abs(f64 a) { return fabs(a); }

// ---- comparisons with Spark semantics: NaN == NaN, NaN is greater than everything ----
template<class T> DEV bool c_eq(T a, T b) { return a == b; }
template<class T> DEV bool c_lt(T a, T b) { return a < b; }
template<class T> DEV bool c_le(T a, T b) { return a <= b; }
template<class T> DEV bool c_gt(T a, T b) { return a > b; }
template<class T> DEV bool c_ge(T a, T b) { return a >= b; }
DEV bool c_eq(bool a, bool b) { return a == b; } DEV bool c_lt(bool a, bool b) { return !a && b; }
DEV bool c_le(bool a, bool b) { return !a || b; } DEV bool c_gt(bool a, bool b) { return a && !b; } DEV bool c_ge(bool a, bool b) { return a || !b; }
DEV bool c_eq(f32 a, f32 b) { return a == b || (a != a && b != b); }
DEV bool c_eq(f64 a, f64 b) { return a == b || (a != a && b != b); }
DEV bool c_lt(f32 a, f32 b) { return (b != b) ? (a == a) : (a < b); }
DEV bool c_lt(f64 a, f64 b) { return (b != b) ? (a == a) : (a < b); }
DEV bool c_le(f32 a, f32 b) { return (b != b) ? true : (a <= b); }
DEV bool c_le(f64 a, f64 b) { return (b != b) ? true : (a <= b); }
DEV bool c_gt(f32 a, f32 b) { return c_lt(b, a); } DEV bool c_gt(f64 a, f64 b) { return c_lt(b, a); }
DEV bool c_ge(f32 a, f32 b) { return c_le(b, a); } DEV bool c_ge(f64 a, f64 b) { return c_le(b, a); }

template<class T> DEV bool is_zero(T a) { return a == 0; }
template<class T> DEV bool is_nan(T a) { return false; }
DEV bool is_nan(f32 a) { return a != a; } DEV bool is_nan(f64 a) { return a != a; }

// ---- division family (callers have already excluded zero divisors) ----
DEV f64 d_div(f64 a, f64 b) { return __ddiv_rn(a, b); }
template<class T> DEV i64 d_idiv(T a, T b) { return (b == (T)-1) ? (i64)(0 - (u64)(i64)a) : ((i64)a / (i64)b); }
template<class T> DEV T d_rem(T a, T b) { return (b == (T)-1) ? (T)0 : (T)(a % b); }
DEV f32 d_rem(f32 a, f32 b) { return fmodf(a, b); } DEV f64 d_rem(f64 a, f64 b) { return fmod(a, b); }
// Spark: r = a % n; if (r < 0) (r + n) % n else r. byte/short are promoted to int; int wraps at 32 bits; long at 64.
template<class T> DEV T d_pmod(T a, T b) {
  i64 r = (b == (T)-1) ? 0 : (i64)((i64)a % (i64)b);
  if (r < 0) {
    i64 s = (sizeof(T) == 8) ? (i64)((u64)r + (u64)(i64)b) : (i64)(i32)((u32)(i32)r + (u32)(i32)b);
    r = (b == (T)-1) ? 0 : (s % (i64)b);
  }
  return (T)r;
}
DEV f32 d_pmod(f32 a, f32 b) { f32 r = fmodf(a, b); return r < 0 ? fmodf(__fadd_rn(r, b), b) : r; }
DEV f64 d_pmod(f64 a, f64 b) { f64 r = fmod(a, b); return r < 0 ? fmod(__dadd_rn(r, b), b) : r; }

// ---- bitwise ----
template<class T> DEV T b_and(T a, T b) { return (T)(a & b); }
template<class T> DEV T b_or(T a, T b) { return (T)(a | b); }
template<class T> DEV T b_xor(T a, T b) { return (T)(a ^ b); }
template<class T> DEV T b_not(T a) { return (T)(~a); }
DEV bool b_and(bool a, bool b) { return a && b; }
DEV i32 s_shl(i32 a, i32 s) { return (i32)((u32)a << (s & 31)); }
DEV i64 s_shl(i64 a, i32 s) { return (i64)((u64)a << (s & 63)); }
DEV i32 s_shr(i32 a, i32 s) { return a >> (s & 31); }
DEV i64 s_shr(i64 a, i32 s) { return a >> (s & 63); }
DEV i32 s_ushr(i32 a, i32 s) { return (i32)((u32)a >> (s & 31)); }
DEV i64 s_ushr(i64 a, i32 s) { return (i64)((u64)a >> (s & 63)); }

// ---- casts ----
DEV i64 fdiv64(i64 a, i64 b) { i64 q = a / b; return ((a % b != 0) && ((a < 0) != (b < 0))) ? q - 1 : q; }
DEV i64 f2i_i64(f64 x) { return x != x ? 0LL : (x >= 9223372036854775808.0 ? 9223372036854775807LL : (x <= -9223372036854775808.0 ? (-9223372036854775807LL - 1) : (i64)x)); }
DEV i32 f2i_i32(f64 x) { return x != x ? 0 : (x >= 2147483648.0 ? 2147483647 : (x <= -2147483648.0 ? (-2147483647 - 1) : (i32)x)); }
DEV int f2i_i16(f64 x) { return norm16(f2i_i32(x)); }
DEV int f2i_i8(f64 x) { return norm8(f2i_i32(x)); }
DEV i64 f2i_i64(f32 x) { return f2i_i64((f64)x); }
DEV i32 f2i_i32(f32 x) { return f2i_i32((f64)x); }
DEV int f2i_i16(f32 x) { return f2i_i16((f64)x); }
DEV int f2i_i8(f32 x) { return f2i_i8((f64)x); }

DEV f64 m_expm1(f64 x) { return __dsub_rn(exp(x), 1.0); }   // spark-rapids computes expm1 as exp(x) - 1
// opaque select: LLVM would turn `a < b ? a : b` into PTX min.f64, whose zero-sign tie handling differs from cuDF's `x < y ? x : y`
DEV f64 sel_f64(bool p, f64 a, f64 b) { f64 r; asm("{ .reg .pred q; setp.ne.u32 q, %3, 0; selp.f64 %0, %1, %2, q; }" : "=d"(r) : "d"(a), "d"(b), "r"((unsigned)p)); return r; }
DEV f32 sel_f32(bool p, f32 a, f32 b) { f32 r; asm("{ .reg .pred q; setp.ne.u32 q, %3, 0; selp.f32 %0, %1, %2, q; }" : "=f"(r) : "f"(a), "f"(b), "r"((unsigned)p)); return r; }
DEV f64 m_cot(f64 x) { return __ddiv_rn(1.0, tan(x)); }   // spark-rapids: 1 / tan(x), no zero-to-null handling
// spark-rapids GpuHypot: x*sqrt(1+(y/x)^2) with x=max(|l|,|r|) (tie -> |r|), y=min, then zero/NaN/inf special cases
DEV f64 m_hypot(f64 l, f64 r) {
  f64 ax = fabs(l), ay = fabs(r);
  bool lg = ax > ay;
  f64 x = lg ? ax : ay, y = lg ? ay : ax;
  f64 q = __ddiv_rn(y, x);
  f64 res = __dmul_rn(x, sqrt(__dadd_rn(1.0, __dmul_rn(q, q))));
  if (x == 0.0 && y == 0.0) res = 0.0;
  if (x != x || y != y) res = __longlong_as_double(0x7ff8000000000000LL);
  if (x == __longlong_as_double(0x7ff0000000000000LL) || y == __longlong_as_double(0x7ff0000000000000LL)) res = __longlong_as_double(0x7ff0000000000000LL);
  return res;
}
DEV f64 m_signum(f64 x) { return x < 0.0 ? -1.0 : (x > 0.0 ? 1.0 : x); }

// ---- date/time (proleptic Gregorian, UTC) ----
// Exact port of libcu++ (cuDF) year_month_day::__from_days / __to_days using 32-bit int/unsigned math, so that
// out-of-range dates (|days| ~ 2^31) wrap exactly like cuDF instead of being "correct".
DEV i32 wi(u32 x) { return (i32)x; }
DEV void civil_from_days(i32 d, i32* py, i32* pm, i32* pd) {
  i32 z = wi((u32)d + 719468u);
  i32 era = (z >= 0 ? z : wi((u32)z - 146096u)) / 146097;
  u32 doe = (u32)z - (u32)era * 146097u;
  u32 yoe = (doe - doe / 1460u + doe / 36524u - doe / 146096u) / 365u;
  i32 yr = wi(yoe + (u32)era * 400u);
  u32 doy = doe - (365u * yoe + yoe / 4u - yoe / 100u);
  u32 mp = (5u * doy + 2u) / 153u;
  u32 dy = doy - (153u * mp + 2u) / 5u + 1u;
  u32 mth = mp + (mp < 10u ? 3u : (u32)-9);
  *py = wi((u32)yr + (mth <= 2u ? 1u : 0u)); *pm = (i32)mth; *pd = (i32)dy;
}
DEV i32 days_from_civil(i32 y0, i32 m, i32 d) {
  i32 y = wi((u32)y0 - (m <= 2 ? 1u : 0u));
  i32 era = (y >= 0 ? y : wi((u32)y - 399u)) / 400;
  u32 yoe = (u32)y - (u32)era * 400u;
  u32 doy = (153u * (u32)(m > 2 ? m - 3 : m + 9) + 2u) / 5u + (u32)d - 1u;
  u32 doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;
  return wi((u32)era * 146097u + doe - 719468u);
}
DEV i64 dt_civil(i32 z) { i32 y, m, d; civil_from_days(z, &y, &m, &d); return ((i64)y << 10) + ((i64)m << 5) + d; }   // packed y/m/d, shared by all date parts
DEV i32 dc_year(i64 p) { return (i32)(i16)(p >> 10); }   // cuDF extract_year yields int16: years beyond +-32767 wrap like cuDF
DEV i32 dc_month(i64 p) { return (i32)((p >> 5) & 31); }
DEV i32 dc_day(i64 p) { return (i32)(p & 31); }
DEV i32 dc_quarter(i64 p) { return (dc_month(p) - 1) / 3 + 1; }
DEV i32 dc_doy(i64 p, i32 z) { return z - days_from_civil(dc_year(p), 1, 1) + 1; }
DEV i32 dc_lastday(i64 p) {
  i32 y = dc_year(p), m = dc_month(p);
  bool leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
  i32 md = (m == 2) ? (leap ? 29 : 28) : ((m == 4 || m == 6 || m == 9 || m == 11) ? 30 : 31);
  return days_from_civil(y, m, md);
}
DEV i32 dt_year(i32 z) { i32 y, m, d; civil_from_days(z, &y, &m, &d); return y; }
DEV i32 dt_month(i32 z) { i32 y, m, d; civil_from_days(z, &y, &m, &d); return m; }
DEV i32 dt_day(i32 z) { i32 y, m, d; civil_from_days(z, &y, &m, &d); return d; }
DEV i32 dt_quarter(i32 z) { i32 y, m, d; civil_from_days(z, &y, &m, &d); return (m - 1) / 3 + 1; }
DEV i32 dt_doy(i32 z) { i32 y, m, d; civil_from_days(z, &y, &m, &d); return z - days_from_civil(y, 1, 1) + 1; }
DEV i32 fmod32(i64 a, i32 b) { i64 r = a % b; return (i32)(r < 0 ? r + b : r); }
// libcu++ weekday::__weekday_from_days (Sunday = 0), 32-bit wrap like cuDF
DEV i32 wd_from_days(i32 d) { i32 t = wi((u32)d + (d >= -4 ? 4u : 5u)); i32 r = t % 7; return d >= -4 ? r : r + 6; }
DEV i32 dt_dow(i32 z) { return wd_from_days(z) + 1; }                 // Sunday=1 .. Saturday=7
DEV i32 dt_weekday(i32 z) { return (wd_from_days(z) + 6) % 7; }       // Monday=0 .. Sunday=6
DEV i32 dt_lastday(i32 z) {
  i32 y, m, d; civil_from_days(z, &y, &m, &d);
  bool leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
  i32 md = (m == 2) ? (leap ? 29 : 28) : ((m == 4 || m == 6 || m == 9 || m == 11) ? 30 : 31);
  return days_from_civil(y, m, md);
}
DEV i32 dt_hour(i64 us) { return fmod32(fdiv64(us, 3600000000LL), 24); }
DEV i32 dt_minute(i64 us) { return fmod32(fdiv64(us, 60000000LL), 60); }
DEV i32 dt_second(i64 us) { return fmod32(fdiv64(us, 1000000LL), 60); }
DEV i32 dt_add(i32 d, i32 n) { return (i32)((u32)d + (u32)n); }
DEV i32 dt_sub(i32 d, i32 n) { return (i32)((u32)d - (u32)n); }
DEV i32 dt_diff(i32 e, i32 s) { return (i32)((u32)e - (u32)s); }

// ---- misc ----
DEV i32 bit_cnt(i64 x) { return (i32)__popcll((u64)x); }
DEV i64 fl_f64(f64 x) { return f2i_i64(x != x ? 0.0 : floor(x)); }
DEV i64 ce_f64(f64 x) { return f2i_i64(x != x ? 0.0 : ceil(x)); }
DEV f64 d_logbase(f64 x, f64 b) { return __ddiv_rn(log(x), log(b)); }


// ---- round / bround on floating point (cuDF half_up / half_even: positive scale splits integer and fractional part) ----
DEV f64 rnd_half_up(f64 e, i32 sc) {
  if (sc == 0) return round(e);
  f64 n = pow(10.0, (f64)(sc < 0 ? -sc : sc));
  if (sc > 0) { f64 ip; f64 fp = modf(e, &ip); return __dadd_rn(ip, __ddiv_rn(round(__dmul_rn(fp, n)), n)); }
  return __dmul_rn(round(__ddiv_rn(e, n)), n);
}
DEV f64 rnd_half_even(f64 e, i32 sc) {
  if (sc == 0) return rint(e);
  f64 n = pow(10.0, (f64)(sc < 0 ? -sc : sc));
  if (sc > 0) { f64 ip; f64 fp = modf(e, &ip); return __dadd_rn(ip, __ddiv_rn(rint(__dmul_rn(fp, n)), n)); }
  return __dmul_rn(rint(__ddiv_rn(e, n)), n);
}
DEV f32 rnd_half_up(f32 e, i32 sc) {
  if (sc == 0) return roundf(e);
  f32 n = (f32)pow(10.0, (f64)(sc < 0 ? -sc : sc));
  if (sc > 0) { f32 ip; f32 fp = modff(e, &ip); return __fadd_rn(ip, __fdiv_rn(roundf(__fmul_rn(fp, n)), n)); }
  return __fmul_rn(roundf(__fdiv_rn(e, n)), n);
}
DEV f32 rnd_half_even(f32 e, i32 sc) {
  if (sc == 0) return rintf(e);
  f32 n = (f32)pow(10.0, (f64)(sc < 0 ? -sc : sc));
  if (sc > 0) { f32 ip; f32 fp = modff(e, &ip); return __fadd_rn(ip, __fdiv_rn(rintf(__fmul_rn(fp, n)), n)); }
  return __fmul_rn(rintf(__fdiv_rn(e, n)), n);
}
DEV f64 rnd_zero_fix(f64 e, f64 r) { return (e != e || e == __longlong_as_double(0x7ff0000000000000LL) || e == __longlong_as_double(0xfff0000000000000LL)) ? e : 0.0; }
DEV f32 rnd_zero_fix(f32 e, f32 r) { return (e != e || e == __int_as_float(0x7f800000) || e == __int_as_float((int)0xff800000)) ? e : 0.0f; }

DEV f32 norm_nz(f32 x) { return x != x ? __int_as_float(0x7fc00000) : (x == 0.0f ? 0.0f : x); }
DEV f64 norm_nz(f64 x) { return x != x ? __longlong_as_double(0x7ff8000000000000LL) : (x == 0.0 ? 0.0 : x); }
"""
}
