"""Расчёты для планирования и анализа пилотного эксперимента."""

import math
from statistics import NormalDist


def n_per_group(sigma, mde, alpha=0.05, power=0.8, comparisons=1):
    """Необходимое число наблюдений в каждой из двух равных групп."""
    z = NormalDist().inv_cdf
    return math.ceil(2 * (z(1 - alpha / (2 * comparisons)) + z(power)) ** 2 * sigma ** 2 / mde ** 2)


def mde_for_n(sigma, n, alpha=0.05, power=0.8, comparisons=1):
    """Минимальный различимый эффект для n наблюдений на группу."""
    z = NormalDist().inv_cdf
    return (2 / n) ** 0.5 * (z(1 - alpha / (2 * comparisons)) + z(power)) * sigma


def diff_ci(t, c, alpha=0.05):
    """Разность средних test − control и нормальный доверительный интервал."""
    d = t.mean() - c.mean()
    se = (t.var() / len(t) + c.var() / len(c)) ** 0.5
    normal = NormalDist()
    q = normal.inv_cdf(1 - alpha / 2)
    p = math.erfc(abs(d / se) / math.sqrt(2)) if se else float(d == 0)
    return {"diff": d, "lo": d - q * se, "hi": d + q * se, "p": p}


def srm_pvalue(sizes):
    """Проверка равных долей для двух или трёх групп."""
    if len(sizes) not in (2, 3):
        raise ValueError("Поддерживаются две или три группы")
    expected = sum(sizes) / len(sizes)
    if expected <= 0:
        raise ValueError("Суммарный размер групп должен быть положительным")
    chi2 = sum((size - expected) ** 2 / expected for size in sizes)
    return math.erfc(math.sqrt(chi2 / 2)) if len(sizes) == 2 else math.exp(-chi2 / 2)


def cuped(y, x):
    """Скорректированная метрика и коэффициент theta."""
    theta = y.cov(x) / x.var()
    return y - theta * (x - x.mean()), theta
