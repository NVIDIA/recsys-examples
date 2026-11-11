"""Random typed expression-tree generator.  A tree renders to (a) a Spark SQL string and (b) a PySpark Column (DataFrame API).

Types: boolean byte short int long float double date timestamp.  Columns come from gen_data.py.
Groups (select with --groups): arith div cmp logic cond nullfn cast bitwise math_exact math_trans date ts
  math_trans = transcendental functions whose bit-exactness vs CPU is NOT expected (v2) - off by default.
"""
import math
import random
import struct

NUM = ["byte", "short", "int", "long", "float", "double"]
INTT = ["byte", "short", "int", "long"]
FP_TYPES = ["float", "double"]
RANK = {t: i for i, t in enumerate(NUM)}
COLS = {
    "boolean": ["b1", "b2"],
    "byte": ["t1", "t2"],
    "short": ["s1", "s2"],
    "int": ["i1", "i2", "i3"],
    "long": ["l1", "l2"],
    "float": ["f1", "f2"],
    "double": ["d1", "d2", "d3"],
    "date": ["dt1", "dt2"],
    "timestamp": ["ts1", "ts2"],
}
import os

HANG_PRONE = (
    os.environ.get("FZ_HANG_PRONE") == "1"
)  # dayofyear(date) hangs spark-rapids 26.08.1 on the 2nd identical run (see KNOWN-DIFFS.md)
ALL_GROUPS = [
    "arith",
    "div",
    "cmp",
    "logic",
    "cond",
    "nullfn",
    "cast",
    "bitwise",
    "math_exact",
    "date",
    "ts",
]
SQLT = {
    "boolean": "BOOLEAN",
    "byte": "TINYINT",
    "short": "SMALLINT",
    "int": "INT",
    "long": "BIGINT",
    "float": "FLOAT",
    "double": "DOUBLE",
    "date": "DATE",
    "timestamp": "TIMESTAMP",
}


def promote(a, b):
    if a == b:
        return a
    return a if RANK[a] > RANK[b] else b


class N:
    """expression node: op, args (list of N), type t, extra (literal value / column name / cast target)"""

    __slots__ = ("op", "args", "t", "x")

    def __init__(self, op, args, t, x=None):
        self.op, self.args, self.t, self.x = op, args, t, x

    def size(self):
        return 1 + sum(a.size() for a in self.args)

    def to_json(self):
        return {
            "op": self.op,
            "t": self.t,
            "x": self.x if not isinstance(self.x, float) else repr(self.x),
            "args": [a.to_json() for a in self.args],
        }

    def copy(self):
        return N(self.op, list(self.args), self.t, self.x)


def from_json(j):
    x = j.get("x")
    if j["op"] == "lit" and j["t"] in FP_TYPES and isinstance(x, str):
        x = float(x)
    return N(j["op"], [from_json(a) for a in j["args"]], j["t"], x)


# ------------------------------------------------------------------ literals
SPECIAL_INT = {
    "byte": [-128, 127, 0, 1, -1, 2, 7, 100],
    "short": [-32768, 32767, 0, 1, -1, 2, 300],
    "int": [-2147483648, 2147483647, 0, 1, -1, 2, 3, 10, 31, 32, 100000, 65536],
    "long": [
        -9223372036854775808,
        9223372036854775807,
        0,
        1,
        -1,
        2,
        1000000,
        4294967296,
        31,
        63,
        64,
    ],
}
SPECIAL_FP = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    3.0,
    10.0,
    1e-3,
    1e3,
    float("nan"),
    float("inf"),
    float("-inf"),
    1.7976931348623157e308,
    -1.7976931348623157e308,
    4.9e-324,
    2.2250738585072014e-308,
    1e15,
    -1e15,
    2.5,
    -2.5,
    3.5,
    1e19,
    -1e19,
    2147483647.0,
    -2147483648.0,
    2147483648.0,
    9.2233720368547758e18,
    3.4028234663852886e38,
    1.401298464324817e-45,
]


def f32(v):
    try:
        return struct.unpack("f", struct.pack("f", v))[0]
    except OverflowError:
        return math.copysign(float("inf"), v)


def gen_lit(rng, t):
    if t == "boolean":
        return N("lit", [], t, rng.random() < 0.5)
    if t in INTT:
        if rng.random() < 0.6:
            return N("lit", [], t, rng.choice(SPECIAL_INT[t]))
        lo, hi = {
            "byte": (-128, 127),
            "short": (-32768, 32767),
            "int": (-(2**31), 2**31 - 1),
            "long": (-(2**63), 2**63 - 1),
        }[t]
        return N(
            "lit",
            [],
            t,
            rng.randint(max(lo, -50), min(hi, 50))
            if rng.random() < 0.5
            else rng.randint(lo, hi),
        )
    if t in FP_TYPES:
        v = (
            rng.choice(SPECIAL_FP)
            if rng.random() < 0.6
            else rng.uniform(-100, 100) * 10 ** rng.randint(-5, 5)
        )
        if t == "float":
            v = f32(v)
        return N("lit", [], t, v)
    if t == "date":
        return N(
            "lit",
            [],
            t,
            rng.choice(
                [
                    "1970-01-01",
                    "2000-02-29",
                    "1969-12-31",
                    "0001-01-01",
                    "9999-12-31",
                    "2024-10-19",
                    "1900-03-01",
                ]
            ),
        )
    if t == "timestamp":
        return N(
            "lit",
            [],
            t,
            rng.choice(
                [
                    "1970-01-01 00:00:00",
                    "2000-02-29 12:34:56.789012",
                    "1969-12-31 23:59:59.999999",
                    "2024-10-19 23:20:00",
                    "0001-01-01 00:00:00",
                ]
            ),
        )
    raise ValueError(t)


def gen_col(rng, t):
    return N("col", [], t, rng.choice(COLS[t]))


# ------------------------------------------------------------------ rendering
def fp_sql(v, t):
    if v != v:
        s = "CAST('NaN' AS DOUBLE)"
    elif v == float("inf"):
        s = "CAST('Infinity' AS DOUBLE)"
    elif v == float("-inf"):
        s = "CAST('-Infinity' AS DOUBLE)"
    elif v == 0 and math.copysign(1, v) < 0:
        s = "CAST('-0.0' AS DOUBLE)"
    else:
        s = f"CAST('{repr(v)}' AS DOUBLE)"
    return f"CAST({s} AS FLOAT)" if t == "float" else s


def sql(n):
    o, a = n.op, n.args
    S = [sql(c) for c in a]
    if o == "col":
        return n.x
    if o == "lit":
        t = n.t
        if t == "boolean":
            return "TRUE" if n.x else "FALSE"
        if t in INTT:
            return f"CAST({n.x} AS {SQLT[t]})"
        if t in FP_TYPES:
            return fp_sql(n.x, t)
        if t == "date":
            return f"DATE'{n.x}'"
        if t == "timestamp":
            return f"TIMESTAMP'{n.x}'"
    if o in (
        "+",
        "-",
        "*",
        "/",
        "%",
        "&",
        "|",
        "^",
        "=",
        "<>",
        "<",
        "<=",
        ">",
        ">=",
        "<=>",
        "AND",
        "OR",
        "div",
    ):
        return f"({S[0]} {o} {S[1]})"
    if o == "neg":
        return f"(-{S[0]})"
    if o == "not":
        return f"(NOT {S[0]})"
    if o == "bitnot":
        return f"(~{S[0]})"
    if o == "isnull":
        return f"({S[0]} IS NULL)"
    if o == "isnotnull":
        return f"({S[0]} IS NOT NULL)"
    if o == "if":
        return f"IF({S[0]}, {S[1]}, {S[2]})"
    if o == "case":
        k = (len(a) - 1) // 2
        parts = " ".join(f"WHEN {S[2*i]} THEN {S[2*i+1]}" for i in range(k))
        return f"(CASE {parts} ELSE {S[-1]} END)"
    if o == "casenoelse":
        k = len(a) // 2
        parts = " ".join(f"WHEN {S[2*i]} THEN {S[2*i+1]}" for i in range(k))
        return f"(CASE {parts} END)"
    if o == "in":
        return f"({S[0]} IN ({', '.join(S[1:])}))"
    if o == "cast":
        return f"CAST({S[0]} AS {SQLT[n.t]})"
    return f"{o}({', '.join(S)})"  # function-call style: abs, pmod, greatest, ...


def df(n, F):
    """PySpark Column.  Uses native API where there is one, F.expr(sql) for the rest."""
    o, a = n.op, n.args
    if o == "col":
        return F.col(n.x)
    if o == "lit":
        t = n.t
        if t == "boolean":
            return F.lit(bool(n.x))
        if t in INTT:
            return F.lit(int(n.x)).cast(t)
        if t in FP_TYPES:
            v = n.x
            return F.lit(float(v)).cast(t)
        return F.expr(sql(n))
    C = [df(c, F) for c in a]
    if o == "+":
        return C[0] + C[1]
    if o == "-":
        return C[0] - C[1]
    if o == "*":
        return C[0] * C[1]
    if o == "/":
        return C[0] / C[1]
    if o == "%":
        return C[0] % C[1]
    if o == "=":
        return C[0] == C[1]
    if o == "<>":
        return C[0] != C[1]
    if o == "<":
        return C[0] < C[1]
    if o == "<=":
        return C[0] <= C[1]
    if o == ">":
        return C[0] > C[1]
    if o == ">=":
        return C[0] >= C[1]
    if o == "<=>":
        return C[0].eqNullSafe(C[1])
    if o == "AND":
        return C[0] & C[1]
    if o == "OR":
        return C[0] | C[1]
    if o == "&":
        return C[0].bitwiseAND(C[1])
    if o == "|":
        return C[0].bitwiseOR(C[1])
    if o == "^":
        return C[0].bitwiseXOR(C[1])
    if o == "neg":
        return -C[0]
    if o == "not":
        return ~C[0]
    if o == "isnull":
        return C[0].isNull()
    if o == "isnotnull":
        return C[0].isNotNull()
    if o == "abs":
        return F.abs(C[0])
    if o == "if":
        return F.when(C[0], C[1]).otherwise(C[2])
    if o == "case":
        k = (len(a) - 1) // 2
        e = F.when(C[0], C[1])
        for i in range(1, k):
            e = e.when(C[2 * i], C[2 * i + 1])
        return e.otherwise(C[-1])
    if o == "casenoelse":
        k = len(a) // 2
        e = F.when(C[0], C[1])
        for i in range(1, k):
            e = e.when(C[2 * i], C[2 * i + 1])
        return e
    if o == "coalesce":
        return F.coalesce(*C)
    if o == "greatest":
        return F.greatest(*C)
    if o == "least":
        return F.least(*C)
    if o == "isnan":
        return F.isnan(C[0])
    if o == "nanvl":
        return F.nanvl(C[0], C[1])
    if o == "cast":
        return C[0].cast(n.t)
    if o == "sqrt":
        return F.sqrt(C[0])
    if o == "in":
        return C[0].isin([df(x, F) for x in a[1:]]) if False else F.expr(sql(n))
    return F.expr(sql(n))


# ------------------------------------------------------------------ generator
class Gen:
    def __init__(self, seed, max_depth=4, groups=None, col_bias=0.7):
        self.rng = random.Random(seed)
        self.max_depth = max_depth
        self.groups = set(groups or ALL_GROUPS)
        self.col_bias = col_bias

    def leaf(self, t):
        return (
            gen_col(self.rng, t)
            if self.rng.random() < self.col_bias
            else gen_lit(self.rng, t)
        )

    def any_type(self, kinds=None):
        r = self.rng.random()
        if kinds:
            return self.rng.choice(kinds)
        if r < 0.08:
            return "boolean"
        if r < 0.14:
            return "byte"
        if r < 0.24:
            return "short"
        if r < 0.50:
            return "int"
        if r < 0.64:
            return "long"
        if r < 0.74:
            return "float"
        return "double"

    def num(self, d, want=None):
        """numeric expression; `want` forces the result type via a cast if needed"""
        e = self._num(d, want)
        if want and e.t != want:
            e = N("cast", [e], want)
        return e

    def _num(self, d, want):
        g, rng = self.groups, self.rng
        t = want or self.any_type(NUM)
        if d <= 0:
            return self.leaf(t)
        choices = []
        if "arith" in g:
            choices += [("arith", 22), ("unary", 6)]
        if "div" in g:
            choices += [("div", 9)]
        if "cond" in g:
            choices += [("if", 8), ("case", 7), ("coalesce", 6), ("greatest", 5)]
        if "nullfn" in g:
            choices += [("nanvl", 2)]
        if "cast" in g:
            choices += [("cast", 10)]
        if "bitwise" in g:
            choices += [("bitwise", 6)]
        if "math_exact" in g:
            choices += [("math", 7)]
        if "math_trans" in g:
            choices += [("trans", 8)]
        if "date" in g:
            choices += [("datefn", 5)]
        if "ts" in g:
            choices += [("tsfn", 3)]
        choices += [("leaf", 5)]
        kinds = [c[0] for c in choices]
        w = [c[1] for c in choices]
        k = rng.choices(kinds, w)[0]
        sub = lambda ty=None: self.num(d - 1, ty) if ty else self.num(d - 1)
        if k == "leaf":
            return self.leaf(t)
        if k == "arith":
            a = sub(want)
            b = sub(want if rng.random() < 0.6 else None)
            if want and a.t != want:
                a = N("cast", [a], want)
            op = rng.choice(["+", "-", "*"])
            return N(op, [a, b], promote(a.t, b.t))
        if k == "unary":
            a = sub(want)
            op = rng.choice(["neg", "abs"])
            return N(op, [a], a.t)
        if k == "div":
            op = rng.choice(["/", "/", "%", "pmod", "div", "div"])
            a = sub()
            b = sub()
            if op == "/":
                return N("/", [a, b], "double")
            if op == "div":
                if a.t in FP_TYPES or b.t in FP_TYPES:
                    a = N("cast", [a], "long")
                    b = N("cast", [b], "long")
                return N("div", [a, b], "long")
            return N(op, [a, b], promote(a.t, b.t))
        if k == "if":
            a = sub(want)
            b = sub(a.t)
            return N("if", [self.boolean(d - 1), a, b], promote(a.t, b.t))
        if k == "case":
            n = rng.randint(2, 4)
            vs = [sub(want) for _ in range(n)]
            ty = vs[0].t
            for v in vs:
                ty = promote(ty, v.t)
            args = []
            for v in vs[:-1] if rng.random() < 0.85 else vs:
                args += [self.boolean(d - 1), v]
            if len(args) == 2 * len(vs):  # no else
                return N("casenoelse", args, ty)
            return N("case", args + [vs[-1]], ty)
        if k in ("coalesce", "greatest"):
            n = rng.randint(2, 3)
            vs = [sub(want) for _ in range(n)]
            ty = vs[0].t
            for v in vs:
                ty = promote(ty, v.t)
            op = (
                rng.choice(["coalesce", "greatest", "least"])
                if k == "greatest"
                else "coalesce"
            )
            return N(op, vs, ty)
        if k == "nanvl":
            ty = want if want in FP_TYPES else rng.choice(FP_TYPES)
            return N("nanvl", [self.num(d - 1, ty), self.num(d - 1, ty)], ty)
        if k == "cast":
            a = sub()
            tgt = want or rng.choice(NUM)
            if rng.random() < 0.1:
                a = self.boolean(d - 1)
            if "date" in g and "ts" in g and rng.random() < 0.15:
                return N(
                    "cast", [self.tsexpr(d - 1)], rng.choice(["long", "double", "int"])
                )  # timestamp -> numeric
            return N("cast", [a], tgt)
        if k == "bitwise":
            ty = want if want in INTT else rng.choice(INTT)
            r = rng.random()
            if r < 0.15:
                return N("bitnot", [self.num(d - 1, ty)], ty)
            if r < 0.65:
                a = self.num(d - 1, ty)
                b = self.num(d - 1, ty)
                return N(rng.choice(["&", "|", "^"]), [a, b], ty)
            if r < 0.9:
                ty = "int" if rng.random() < 0.6 else "long"
                op = rng.choice(["shiftleft", "shiftright", "shiftrightunsigned"])
                return N(op, [self.num(d - 1, ty), self.num(d - 1, "int")], ty)
            return N("bit_count", [self.num(d - 1, ty)], "int")
        if k == "math":
            op = rng.choice(
                ["sqrt", "floor", "ceil", "signum", "rint", "degrees", "radians"]
            )
            a = self.num(d - 1, "double" if rng.random() < 0.7 else None)
            ty = "long" if op in ("floor", "ceil") else "double"
            return N(op, [a], ty)
        if k == "trans":
            op = rng.choice(
                [
                    "sin",
                    "cos",
                    "tan",
                    "exp",
                    "log",
                    "log2",
                    "log10",
                    "cbrt",
                    "atan",
                    "tanh",
                    "expm1",
                    "log1p",
                    "pow",
                    "atan2",
                    "hypot",
                ]
            )
            if op in ("pow", "atan2", "hypot"):
                return N(
                    op, [self.num(d - 1, "double"), self.num(d - 1, "double")], "double"
                )
            return N(op, [self.num(d - 1, "double")], "double")
        if k == "datefn":
            op = rng.choice(
                [
                    "year",
                    "month",
                    "dayofmonth",
                    "dayofweek",
                    "quarter",
                    "weekday",
                    "datediff",
                ]
                + (["dayofyear"] if HANG_PRONE else [])
            )
            if op == "datediff":
                return N(op, [self.dateexpr(d - 1), self.dateexpr(d - 1)], "int")
            return N(op, [self.dateexpr(d - 1)], "int")
        if k == "tsfn":
            op = rng.choice(["hour", "minute", "second"])
            return N(op, [self.tsexpr(d - 1)], "int")
        return self.leaf(t)

    def dateexpr(self, d):
        rng = self.rng
        if d <= 0 or rng.random() < 0.5:
            return self.leaf("date")
        k = rng.choice(["date_add", "date_sub", "last_day", "cast_ts", "if"])
        if k in ("date_add", "date_sub"):
            return N(
                k,
                [
                    self.dateexpr(d - 1),
                    self.num(d - 1, rng.choice(["byte", "short", "int"])),
                ],
                "date",
            )
        if k == "last_day":
            return N(k, [self.dateexpr(d - 1)], "date")
        if k == "cast_ts" and "ts" in self.groups:
            return N("cast", [self.tsexpr(d - 1)], "date")
        return N(
            "if",
            [self.boolean(d - 1), self.dateexpr(d - 1), self.dateexpr(d - 1)],
            "date",
        )

    def tsexpr(self, d):
        rng = self.rng
        if d <= 0 or rng.random() < 0.6:
            return self.leaf("timestamp")
        k = rng.choice(["if", "cast_date", "cast_long"])
        if k == "cast_date":
            return N("cast", [self.dateexpr(d - 1)], "timestamp")
        if k == "cast_long":
            return N(
                "cast",
                [self.num(d - 1, rng.choice(["int", "long", "short"]))],
                "timestamp",
            )
        return N(
            "if",
            [self.boolean(d - 1), self.tsexpr(d - 1), self.tsexpr(d - 1)],
            "timestamp",
        )

    def boolean(self, d):
        g, rng = self.groups, self.rng
        if d <= 0:
            return self.leaf("boolean")
        choices = []
        if "cmp" in g:
            choices += [("cmp", 22), ("nulleq", 4), ("in", 4)]
        if "logic" in g:
            choices += [("and", 9), ("or", 9), ("not", 4)]
        if "nullfn" in g:
            choices += [("isnull", 6), ("isnan", 3)]
        if "cast" in g:
            choices += [("castb", 3)]
        if "cond" in g:
            choices += [("ifb", 3)]
        choices += [("leaf", 3)]
        k = rng.choices([c[0] for c in choices], [c[1] for c in choices])[0]
        if k == "leaf":
            return self.leaf("boolean")
        if k in ("cmp", "nulleq"):
            r = rng.random()
            if r < 0.78:
                a = self.num(d - 1)
                b = self.num(d - 1)
            elif r < 0.88:
                a, b = self.dateexpr(d - 1), self.dateexpr(d - 1)
            elif r < 0.95:
                a, b = self.tsexpr(d - 1), self.tsexpr(d - 1)
            else:
                a, b = self.boolean(d - 1), self.boolean(d - 1)
            op = (
                "<=>"
                if k == "nulleq"
                else rng.choice(["=", "<>", "<", "<=", ">", ">="])
            )
            return N(op, [a, b], "boolean")
        if k == "in":
            ty = self.any_type(NUM)
            a = self.num(d - 1, ty)
            return N(
                "in",
                [a] + [gen_lit(rng, ty) for _ in range(rng.randint(1, 4))],
                "boolean",
            )
        if k == "and":
            return N("AND", [self.boolean(d - 1), self.boolean(d - 1)], "boolean")
        if k == "or":
            return N("OR", [self.boolean(d - 1), self.boolean(d - 1)], "boolean")
        if k == "not":
            return N("not", [self.boolean(d - 1)], "boolean")
        if k == "isnull":
            a = self.num(d - 1) if rng.random() < 0.7 else self.boolean(d - 1)
            return N(rng.choice(["isnull", "isnotnull"]), [a], "boolean")
        if k == "isnan":
            return N("isnan", [self.num(d - 1, rng.choice(FP_TYPES))], "boolean")
        if k == "castb":
            return N("cast", [self.num(d - 1)], "boolean")
        if k == "ifb":
            return N(
                "if",
                [self.boolean(d - 1), self.boolean(d - 1), self.boolean(d - 1)],
                "boolean",
            )
        return self.leaf("boolean")

    def top(self):
        """one projected expression: mostly numeric, some boolean, some date/timestamp"""
        r = self.rng.random()
        d = self.rng.randint(1, self.max_depth)
        if r < 0.65:
            return self.num(d)
        if r < 0.85:
            return self.boolean(d)
        if r < 0.93 and "date" in self.groups:
            return self.dateexpr(d)
        if "ts" in self.groups:
            return self.tsexpr(d)
        return self.num(d)


def render_sql(n):
    return sql(n)


def ops_in(n):
    s = {n.op}
    for a in n.args:
        s |= ops_in(a)
    return s


if __name__ == "__main__":
    g = Gen(1)
    for _ in range(8):
        e = g.top()
        print(e.t, e.size(), sql(e))
