"""
STATS 531 Midterm Project
ARMA(1,1) vs SARIMA(1,0,1)(1,0,1)_4 Model Comparison
Google (GOOG) Weekly Realized Volatility

This script fits both models, compares AIC, and runs full residual diagnostics.
Adjust the data loading section to match your data format.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import jarque_bera
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SECTION 1: DATA LOADING
# ============================================================
# >>> MODIFY THIS SECTION TO LOAD YOUR DATA <<<
# Your data should be a pandas Series called `log_rv` indexed by date
# containing the log of weekly realized volatility for GOOG
#
# Example:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# log_rv = np.log(df['realized_vol'])
#
# For now, here's a placeholder that you should replace:

import yfinance as yf
goog = yf.download('GOOG', start='2015-01-01', end='2025-01-01')
log_returns = np.log(goog['Close'] / goog['Close'].shift(1)).dropna()
rv_weekly = log_returns.rolling(window=5).std() * np.sqrt(252)
rv_weekly = rv_weekly.iloc[4::5].dropna()  # non-overlapping weekly windows
log_rv = np.log(rv_weekly)
log_rv.name = 'log_rv'


print(f"\nSeries length: {len(log_rv)} observations")
print(f"Date range: {log_rv.index[0].date()} to {log_rv.index[-1].date()}")
# print(f"Mean: {log_rv.mean():.4f}, Std: {log_rv.std():.4f}\n")


# ============================================================
# SECTION 2: MODEL FITTING
# ============================================================

# --- Model 1: ARMA(1,1) ---
print("-" * 60)
print("Fitting ARMA(1,1)...")
print("-" * 60)
arma_model = SARIMAX(log_rv, order=(1, 0, 1), trend='c',
                     enforce_stationarity=True,
                     enforce_invertibility=True)
arma_result = arma_model.fit(disp=False)
print(arma_result.summary())

# --- Model 2: SARIMA(1,0,1)(1,0,1)_4 ---
# s=4 because spectral peak at f=0.24 ≈ 1/4.2 weeks (monthly cycle)
print("\n" + "-" * 60)
print("Fitting SARIMA(1,0,1)(1,0,1)_4...")
print("-" * 60)
sarima_model = SARIMAX(log_rv, order=(1, 0, 1),
                       seasonal_order=(1, 0, 1, 4), trend='c',
                       enforce_stationarity=True,
                       enforce_invertibility=True)
sarima_result = sarima_model.fit(disp=False)
print(sarima_result.summary())


# ============================================================
# SECTION 3: MODEL COMPARISON SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"{'Model':<30} {'AIC':>10} {'BIC':>10} {'LogLik':>10} {'Params':>8}")
print("-" * 68)
print(f"{'ARMA(1,1)':<30} {arma_result.aic:>10.2f} {arma_result.bic:>10.2f} "
      f"{arma_result.llf:>10.2f} {arma_result.df_model:>8}")
print(f"{'SARIMA(1,0,1)(1,0,1)_4':<30} {sarima_result.aic:>10.2f} {sarima_result.bic:>10.2f} "
      f"{sarima_result.llf:>10.2f} {sarima_result.df_model:>8}")
print("-" * 68)
aic_diff = arma_result.aic - sarima_result.aic
if aic_diff > 2:
    print(f"SARIMA is preferred (ΔAIC = {aic_diff:.2f})")
elif aic_diff < -2:
    print(f"ARMA is preferred (ΔAIC = {abs(aic_diff):.2f})")
else:
    print(f"Models are statistically equivalent (ΔAIC = {abs(aic_diff):.2f} < 2)")
    print("Prefer ARMA(1,1) by parsimony principle.")


# ============================================================
# SECTION 4: RESIDUAL EXTRACTION
# ============================================================

arma_resid = arma_result.resid
sarima_resid = sarima_result.resid

# Standardize residuals for QQ plot comparability
arma_std_resid = (arma_resid - arma_resid.mean()) / arma_resid.std()
sarima_std_resid = (sarima_resid - sarima_resid.mean()) / sarima_resid.std()


# ============================================================
# SECTION 5: DIAGNOSTIC TESTS
# ============================================================

def run_diagnostics(resid, model_name):
    """Run full suite of residual diagnostic tests."""
    print(f"\n{'=' * 60}")
    print(f"RESIDUAL DIAGNOSTICS: {model_name}")
    print(f"{'=' * 60}")

    # --- Ljung-Box Test (no autocorrelation in residuals) ---
    print(f"\n--- Ljung-Box Test (H0: residuals are white noise) ---")
    lb_lags = [10, 15, 20]
    lb_results = acorr_ljungbox(resid, lags=lb_lags, return_df=True)
    for lag in lb_lags:
        stat = lb_results.loc[lag, 'lb_stat']
        pval = lb_results.loc[lag, 'lb_pvalue']
        verdict = "✓ PASS (white noise)" if pval > 0.05 else "✗ FAIL (autocorrelation remains)"
        print(f"  Lag {lag:>2}: Q = {stat:>8.3f}, p = {pval:.4f}  {verdict}")

    # --- Jarque-Bera Test (normality) ---
    jb_stat, jb_pval, skew, kurt = jarque_bera(resid)
    jb_verdict = "✓ PASS (normal)" if jb_pval > 0.05 else "✗ FAIL (non-normal)"
    print(f"\n--- Jarque-Bera Test (H0: residuals are normal) ---")
    print(f"  JB = {jb_stat:.3f}, p = {jb_pval:.4f}  {jb_verdict}")
    print(f"  Skewness = {skew:.4f}, Kurtosis = {kurt:.4f}")

    # --- ARCH LM Test (no conditional heteroskedasticity) ---
    print(f"\n--- ARCH LM Test (H0: no ARCH effects / constant variance) ---")
    from statsmodels.stats.diagnostic import het_arch
    arch_lags = [5, 10]
    for lag in arch_lags:
        arch_stat, arch_pval, _, _ = het_arch(resid, nlags=lag)
        arch_verdict = "✓ PASS (homoskedastic)" if arch_pval > 0.05 else "✗ FAIL (ARCH effects present)"
        print(f"  Lag {lag:>2}: LM = {arch_stat:>8.3f}, p = {arch_pval:.4f}  {arch_verdict}")

    # --- Shapiro-Wilk Test (normality, more powerful for small samples) ---
    sw_stat, sw_pval = stats.shapiro(resid[:500])  # Shapiro-Wilk limited to 5000
    sw_verdict = "✓ PASS (normal)" if sw_pval > 0.05 else "✗ FAIL (non-normal)"
    print(f"\n--- Shapiro-Wilk Test (H0: residuals are normal) ---")
    print(f"  W = {sw_stat:.4f}, p = {sw_pval:.4f}  {sw_verdict}")

    return lb_results


arma_lb = run_diagnostics(arma_resid, "ARMA(1,1)")
sarima_lb = run_diagnostics(sarima_resid, "SARIMA(1,0,1)(1,0,1)_4")


# ============================================================
# SECTION 6: RESIDUAL COMPARISON PLOT
# ============================================================

fig = plt.figure(figsize=(18, 20))
fig.suptitle('Residual Diagnostics: ARMA(1,1) vs SARIMA(1,0,1)(1,0,1)$_4$',
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(5, 2, hspace=0.4, wspace=0.3,
                       top=0.95, bottom=0.04, left=0.08, right=0.96)

models = [
    ('ARMA(1,1)', arma_resid, arma_std_resid, '#2196F3'),
    ('SARIMA(1,0,1)(1,0,1)$_4$', sarima_resid, sarima_std_resid, '#E91E63')
]

for col, (name, resid, std_resid, color) in enumerate(models):

    # Row 1: Residual Time Series
    ax1 = fig.add_subplot(gs[0, col])
    ax1.plot(resid.index, resid.values, color=color, alpha=0.7, linewidth=0.8)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_title(f'{name} — Residuals Over Time', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Residual')
    ax1.tick_params(axis='x', rotation=30)

    # Row 2: ACF of Residuals
    ax2 = fig.add_subplot(gs[1, col])
    plot_acf(resid, ax=ax2, lags=30, alpha=0.05,
             color=color, vlines_kwargs={'colors': color})
    ax2.set_title(f'{name} — ACF of Residuals', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Lag')

    # Row 3: PACF of Residuals
    ax3 = fig.add_subplot(gs[2, col])
    plot_pacf(resid, ax=ax3, lags=30, alpha=0.05, method='ywm',
              color=color, vlines_kwargs={'colors': color})
    ax3.set_title(f'{name} — PACF of Residuals', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Lag')

    # Row 4: QQ Plot
    ax4 = fig.add_subplot(gs[3, col])
    qq = stats.probplot(std_resid, dist="norm")
    ax4.scatter(qq[0][0], qq[0][1], alpha=0.5, s=15, color=color, edgecolors='none')
    ax4.plot(qq[0][0], qq[1][0] * qq[0][0] + qq[1][1],
             color='black', linewidth=1.2, linestyle='--', label='Normal reference')
    ax4.set_title(f'{name} — QQ Plot', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Theoretical Quantiles')
    ax4.set_ylabel('Sample Quantiles')
    ax4.legend(fontsize=9)

    # Row 5: Histogram of Residuals
    ax5 = fig.add_subplot(gs[4, col])
    ax5.hist(std_resid, bins=35, density=True, alpha=0.6, color=color, edgecolor='white')
    x_range = np.linspace(std_resid.min() - 0.5, std_resid.max() + 0.5, 200)
    ax5.plot(x_range, stats.norm.pdf(x_range), 'k--', linewidth=1.5, label='Normal PDF')
    ax5.set_title(f'{name} — Residual Distribution', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Standardized Residual')
    ax5.set_ylabel('Density')
    ax5.legend(fontsize=9)

plt.savefig('/Users/huyle/Documents/GitHub/STATS531/project/residual_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n\nResidual comparison plot saved: residual_comparison.png")


# ============================================================
# SECTION 7: PARAMETER COMPARISON TABLE
# ============================================================

print(f"\n{'=' * 60}")
print("PARAMETER ESTIMATES")
print(f"{'=' * 60}")

print(f"\nARMA(1,1):")
for param, value in arma_result.params.items():
    pval = arma_result.pvalues[param]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {param:<20} = {value:>8.4f}  (p = {pval:.4f}) {sig}")

print(f"\nSARIMA(1,0,1)(1,0,1)_4:")
for param, value in sarima_result.params.items():
    pval = sarima_result.pvalues[param]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {param:<20} = {value:>8.4f}  (p = {pval:.4f}) {sig}")

# Check if seasonal terms are significant
print(f"\n{'=' * 60}")
print("SEASONAL TERM SIGNIFICANCE CHECK")
print(f"{'=' * 60}")
seasonal_params = [p for p in sarima_result.params.index if 'seasonal' in p.lower() or '.S.' in p]
if not seasonal_params:
    # Try alternative naming
    all_params = sarima_result.params.index.tolist()
    seasonal_params = [p for p in all_params if 'ar.S' in p or 'ma.S' in p
                       or 'ar.L4' in p or 'ma.L4' in p]

if seasonal_params:
    for param in seasonal_params:
        pval = sarima_result.pvalues[param]
        verdict = "SIGNIFICANT" if pval < 0.05 else "NOT significant"
        print(f"  {param}: p = {pval:.4f} → {verdict}")
else:
    print("  Seasonal parameters found in model:")
    for p in sarima_result.params.index:
        print(f"    {p}: {sarima_result.params[p]:.4f} (p={sarima_result.pvalues[p]:.4f})")

print(f"\n{'=' * 60}")
print("INTERPRETATION GUIDE")
print(f"{'=' * 60}")
print("""
Key questions to answer in your paper:

1. AIC COMPARISON: Is the SARIMA AIC lower by >2 than ARMA?
   - If yes: seasonal terms capture real structure → use SARIMA
   - If no: seasonal terms add complexity without benefit → use ARMA(1,1)

2. SEASONAL PARAMETERS: Are the seasonal AR/MA coefficients significant?
   - If p > 0.05: the 0.24 spectral peak was noise, not exploitable seasonality
   - If p < 0.05: monthly cycle exists but check if AIC actually improves

3. LJUNG-BOX: Do both models produce white noise residuals?
   - Both pass: prefer simpler ARMA(1,1)
   - Only SARIMA passes: seasonal terms are needed
   - Both fail: neither model is adequate, consider GARCH or higher orders

4. ARCH LM TEST: Do residuals show conditional heteroskedasticity?
   - If yes: volatility-of-volatility exists → mention GARCH as future work
   - If no: ARMA variance assumptions are satisfied

5. QQ PLOT: Check the tails.
   - Fat tails are common for financial vol series
   - Report honestly; doesn't invalidate ARMA but affects prediction intervals
""")

print("Done! Check residual_comparison.png for the visual comparison.")