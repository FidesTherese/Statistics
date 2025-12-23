# Mean
def mean(dt: list):
    n = len(dt)
    summ = 0
    for i in range(n):
        summ += dt[i]
    result = summ / n
    return result

# Variance
def variance(dt: list, type: str = 'sample'):
    dt_mean = mean(dt)
    n = len(dt)

    if type == 'sample':
        summ_diff = 0
        for i in range(n):
            summ_diff += (dt[i] - dt_mean)**2
        result = summ_diff / (n - 1)
        return result
    elif type == 'population':
        summ_diff = 0
        for i in range(n):
            summ_diff += (dt[i] - dt_mean)**2
        result = summ_diff / n
        return result
    else:
        raise ValueError('Only sample and population are valid type!')

# Standard Deviation
def std(dt: list, type: str = 'sample'):
    from math import sqrt

    if type == 'sample':
        result = sqrt(variance(dt, type))
        return result
    elif type == 'population':
        result = sqrt(variance(dt, type))
        return result
    else:
        raise ValueError('Only sample and population are valid type!')

# Median
def median(dt: list):
    n = len(dt)
    dt = sorted(dt)

    # When n is odd number
    if n % 2 != 0:
        position = int(((n + 1) / 2) - 1)
        median = dt[position]
        return median
    # When n is even number
    else:
        position_left = int((n / 2) - 1)
        position_right = int(position_left + 1)
        median = (dt[position_left] + dt[position_right]) / 2
        return median

# Quatiles
def quatiles(dt: list):
    n = len(dt)
    dt = sorted(dt)

    # When n is odd number
    if n % 2 != 0:
        position = int(((n + 1) / 2) - 1)
        dt_split_left = dt[0: position]
        dt_split_right = dt[position + 1: n]
        # 25% quatiles
        quatile_upper = median(dt_split_left)
        # 75% quatiles
        quatile_lower = median(dt_split_right)
        # Return
        return quatile_upper, quatile_lower
    # When n is even number
    else:
        position_left = int((n / 2) - 1)
        position_right = int(position_left + 1)
        dt_split_left = dt[0: position_left + 1]
        dt_split_right = dt[position_right: n]
        # 25% quatiles
        quatile_upper = median(dt_split_left)
        # 75% quatiles
        quatile_lower = median(dt_split_right)
        # Return
        return quatile_upper, quatile_lower

# Ranking
def get_rank(dt: list):
    sorted_dt = sorted([(val, i) for i, val in enumerate(dt)])
    ranks = [0] * len(dt)

    i = 0
    while i < len(sorted_dt):
        j = i
        while (j < len(sorted_dt) - 1) and (sorted_dt[j+1][0] == sorted_dt[j][0]):
            j += 1

        avg_rank = (i + j + 2) / 2
        for k in range(i, j + 1):
            ranks[sorted_dt[k][1]] = avg_rank
        i = j + 1
        
    return ranks

# Mode
def mode(dt: list):
    from collections import Counter
    counts = Counter(dt)
    max_count = max(counts, key = counts.get)

    return max_count

# Range
def dt_range(dt: list):
    dt = sorted(dt)
    maximum = dt[-1]
    minimum = dt[0]
    result = maximum - minimum

    return result

# Coefficient of Variance
def coefficient_of_variance(dt: list, type: str):
    # Std if sample
    if type == 'sample':
        sd = std(dt, type)
    elif type == 'population':
        sd = std(dt, type)
    else:
        raise ValueError('Only sample and population are valid type!')

    # Mean
    dt_mean = mean(dt)

    # CoV
    CoV = sd / dt_mean

    return CoV

# Gamma Function
def gamma_function(z: float):
    import numpy as np
    from scipy.integrate import quad

    if z <= 0:
        raise ValueError('Parameter z must be greater than zero!')

    result, error = quad(
            lambda t, z: (t**(z - 1)) * np.exp(-t),
            0,
            np.inf,
            args = (z,)
    )

    return result

# T-distribution PDF
def t_pdf(t_value: float, df: float):
    from math import pi, sqrt
    PDF_t = gamma_function((df + 1) / 2) / (sqrt(pi * df) * gamma_function(df / 2)) * (1 + (t_value**2 / df))**(-(df + 1) / 2)

    return PDF_t

# Pearson Correlation Coefficient
def pearson_corr(x: list, y: list):
    from math import sqrt, inf
    from scipy.integrate import quad
    x_mean = mean(x)
    y_mean = mean(y)

    # Check length
    if len(x) != len(y):
        raise ValueError('Length of both lists must be equal!')

    # Numerator
    numerator = 0
    for i in range(len(x)):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)

    # Squared Deviation of x
    x_sq_deviation = 0
    for k in range(len(x)):
        x_sq_deviation += (x[k] - x_mean)**2

    # Squared Deviation of y
    y_sq_deviation = 0
    for l in range(len(y)):
        y_sq_deviation += (y[l] - y_mean)**2

    # Result
    denominator = sqrt(x_sq_deviation * y_sq_deviation)
    if denominator == 0:
        result = 0.0
    else:
        result = numerator / denominator

    # Significant
    if abs(result) >= 1.0:
        t_stat = 1e15 # To avoid zero division error
    else:
        t_stat = (result * sqrt(len(x) - 2)) / sqrt(1 - result**2)

    df = len(x) - 2
    p_val_tail, _ = quad(t_pdf_log, abs(t_stat), inf, args = (df,))
    sig_xy = 2 * p_val_tail

    return result, sig_xy

# F-distribution PDF
def f_PDF(f_stat: float, df_between: float, df_within: float):
    from math import sqrt
    
    a = df_between / 2
    b = df_within / 2
    beta = (gamma_function(a) * gamma_function(b)) / gamma_function(a + b)

    if f_stat <= 0:
        return 0
    else:
        numerator = sqrt(((df_between * f_stat)**df_between * df_within**df_within) / (df_between * f_stat + df_within)**(df_between + df_within))
        denominator = f_stat * beta
        PDF_f = numerator / denominator
        return PDF_f

# ANOVA
def anova(*groups):
    from math import sqrt, inf
    from scipy.integrate import quad

    # 1. Preparation
    # Numbers of groups
    k = len(groups)
    # Flatten
    all_data = []
    for g in groups:
        all_data.extend(g)
    # Numbers of all data
    n_total = len(all_data)
    # Grand mean
    summ_all_data = 0
    for i in range(n_total):
        summ_all_data += all_data[i]
    grand_mean = summ_all_data / n_total

    # 2. SSB
    group_means = [sum(g) / len(g) for g in groups]
    ssb = sum(len(g) * (m - grand_mean)**2 for g, m in zip(groups, group_means))

    # 3. SSW
    ssw = 0
    for i, g in enumerate(groups):
        group_mean = group_means[i]
        ssw += sum((x - group_mean)**2 for x in g)

    # 4. F-statistic
    df_between = k - 1
    df_within = n_total - k
    msb = ssb / df_between
    msw = ssw / df_within
    f_stat = msb / msw

    # 5. Significant
    p_value, _ = quad(f_PDF, f_stat, inf, args = (df_between, df_within,))

    return f_stat, p_value

# Chi-Square Distribution PDF
def chi_square_pdf(x: float, df: float):
    from math import exp

    # Check the length of data (for numerical integration)
    if x <= 0:
        return 0
    
    # Define PDF
    k = df
    term_1 = 1 / (2**(k/2) * gamma_function(k/2))
    term_2 = x**(k/2 - 1)
    term_3 = exp(-x / 2)
    chi_pdf = term_1 * term_2 * term_3

    return chi_pdf

# Chi-Square Test
def chi_square_test(observed: list, expected: list):
    from math import inf
    from scipy.integrate import quad

    chi_stat = 0
    for o, e in zip(observed, expected):
        chi_stat += (o - e)**2 / e
    
    df = len(observed) - 1

    p_value, _ = quad(chi_square_pdf, chi_stat, inf, args = (df,))

    return p_value

# Logarithmic Gamma Function
def log_gamma_function(z: float):
    import numpy as np
    if z <= 0:
        raise ValueError('Parameter z must be greater than zero.')
    
    # Stirling Approximation
    # When z is too small
    if z < 7:
        return log_gamma_function(z + 1) - np.log(z)
    else:
        log_2pi = np.log(2 * np.pi)
        log_gamma = 0.5 * log_2pi + (z - 0.5) * np.log(z) - z + (1 / (12 * z)) - (1 / (360 * z**3))
        return log_gamma
    
# Logarithmic T-Distribution PDF
def t_pdf_log(t: float, df: float):
    import numpy as np
    ln_numerator = log_gamma_function((df + 1) / 2)
    ln_denominator = log_gamma_function(df / 2) + 0.5 * np.log(df * np.pi)
    ln_term = -((df + 1) / 2) * np.log(1 + (t**2 / df))
    ln_t_pdf = ln_numerator - ln_denominator + ln_term
    
    return np.exp(ln_t_pdf)

# Multiple Linear Regression
def lm(X, Y): # Where X is a matrix and Y is a vector, using NDArray format
    import numpy as np
    from scipy.integrate import quad
    from math import inf
    
    # 1. Design matrix
    n_samples = X.shape[0]
    X_design = np.hstack([np.ones((n_samples, 1)), X])

    # 2. Beta
    xtx = X_design.T @ X_design
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ X_design.T @ Y

    # 3. Significance
    # Prediction and Residuals
    Y_pred = X_design @ beta
    residuals = Y - Y_pred
    # Residual Variance
    df = n_samples - X_design.shape[1]
    sigma_sq = np.sum(residuals**2) / df
    # Covariance Matrix of Coefficient
    cov_beta = sigma_sq * xtx_inv
    se_beta = np.sqrt(np.diag(cov_beta)).reshape(-1, 1)
    # T-statistic and P-value
    t_stat = beta / se_beta
    p_value = [2 * quad(t_pdf_log, abs(t), inf, args = (df,))[0] for t in t_stat]

    # 4. R-Squared
    r_sq = 1 - (np.sum(residuals**2) / np.sum(Y - np.mean(Y)**2))
    # As the inverse of a matrix is used, sometime the result will be out of definition
    if r_sq > 1:
        r_sq = 1
    elif r_sq < -1:
        r_sq = -1
    else:
        pass

    # 5. Return
    return {
        'beta': beta.flatten(),
        'p': p_value,
        'se': se_beta.flatten(),
        't': t_stat.flatten(),
        'df': df,
        'rsq': r_sq
    }

# Levene's Homoscedasticity Test
def levene_homoscedasticity_test(*groups, center = 'median'):
    import numpy as np
    z_groups = []
    for g in groups:
        g = np.array(g)
        if center == 'mean':
            c = np.mean(g)
        elif center == 'median':
            c = np.median(g)
        else:
            raise ValueError('Only mean and median are valid type!')
        
        z_g = np.abs(g - c)
        z_groups.append(z_g.tolist())

    anova_f_stat, anova_p_value = anova(*z_groups)

    return {
            'levene_f': anova_f_stat,
            'levene_p': anova_p_value
        }

# T-test
def t_test(dt1: list = [], dt2 = [], type: str = 'no', mu: float = 0):
    def one_way_t_test(dt, mu):
        from math import sqrt, inf
        from scipy.integrate import quad

        dt_mean = mean(dt)
        numerator = dt_mean - mu
        se = std(dt) / sqrt(len(dt))
        t_stat = numerator / se
        df = len(dt) - 1

        p_val_tail, _ = quad(t_pdf_log, abs(t_stat), inf, args = (df,))
        p_value = 2 * p_val_tail

        return {
            'sample_mean': dt_mean,
            't': t_stat,
            'p': p_value,
            'df': df,
        }
    
    def independent_t_test(dt1: list, dt2: list):
        levene_result = levene_homoscedasticity_test(dt1, dt2)
        if levene_result['levene_p'] < 0.05:
            # Data is not homodastic, then use Welth's t-test instead
            from math import sqrt, inf
            from scipy.integrate import quad

            dt1_mean = mean(dt1)
            dt2_mean = mean(dt2)
            dt1_var = variance(dt1)
            dt2_var = variance(dt2)
            dt1_n = len(dt1)
            dt2_n = len(dt2)
            
            t_stat = (dt1_mean - dt2_mean) / (sqrt(dt1_var/dt1_n + dt2_var/dt2_n))
            numerator_df = (dt1_var/dt1_n + dt2_var/dt2_n)**2
            denominator_df = ((dt1_var/dt1_n)**2 / (dt1_n - 1)) + ((dt2_var/dt2_n)**2 / (dt2_n - 1))
            df = numerator_df / denominator_df

            p_val_tail, _ = quad(t_pdf_log, abs(t_stat), inf, args = (df,))
            p_value = 2 * p_val_tail

            return {
                't': t_stat,
                'p': p_value,
                'df': df,
            }
        else:
            # Data is homodastic, then use Student's t-test
            from math import sqrt, inf
            from scipy.integrate import quad

            dt1_mean = mean(dt1)
            dt2_mean = mean(dt2)
            dt1_var = variance(dt1)
            dt2_var = variance(dt2)
            dt1_n = len(dt1)
            dt2_n = len(dt2)
            var_p = ((dt1_n - 1)*dt1_var + (dt2_n - 1)*dt2_var) / (dt1_n + dt2_n - 2)

            t_stat = (dt1_mean - dt2_mean) / (sqrt(var_p*(1/dt1_n + 1/dt2_n)))
            df = dt1_n + dt2_n - 2

            p_val_tail, _ = quad(t_pdf_log, abs(t_stat), inf, args = (df,))
            p_value = 2 * p_val_tail

            return {
                't': t_stat,
                'p': p_value,
                'df': df
            }
        
    def paired_t_test(dt1: list, dt2: list):
        if len(dt1) != len(dt2):
            raise ValueError('Length of two variables must be equal when applying paired t-test!')
        
        diff = [a - b for a, b in zip(dt1, dt2)]

        result = one_way_t_test(diff, mu = 0)
        
        return result
    
    def yuan_welch_t_test(dt1: list, dt2: list, gamma: float):
        import numpy as np
        from scipy.stats import t

        dt1 = np.array(dt1)
        dt2 = np.array(dt2)

        dt1_copy = np.sort(dt1.copy())
        dt2_copy = np.sort(dt2.copy())

        # Define the size of data
        n1 = dt1.size
        n2 = dt2.size

        # Calculate how many data we trim
        g1 = int(gamma * n1)
        g2 = int(gamma * n2)

        # Trimmed data
        dt1 = np.sort(dt1)[g1: n1 - g1]
        dt2 = np.sort(dt2)[g2: n2 - g2]

        # Define effective number of samples
        h1 = n1 - 2 * g1
        h2 = n2 - 2 * g2

        # Define winsorized data
        def winsorize(x, gamma):
            x = np.array(x)
            n = len(x)
            g = int(np.floor(gamma * n))

            x_w = x.copy()
            if g > 0:
                x_w[:g] = x[g]
                x_w[-g:] = x[-g - 1]

            return x_w

        # Winsorized variance
        v1 = np.var(winsorize(dt1_copy, gamma), ddof=1)
        v2 = np.var(winsorize(dt2_copy, gamma), ddof=1)
        s1_sq = v1 / (1 - 2 * gamma) ** 2
        s2_sq = v2 / (1 - 2 * gamma) ** 2

        # Trimmed mean
        x1 = np.mean(dt1)
        x2 = np.mean(dt2)

        # Welch–Satterthwaite degrees of freedom
        numerator = (s1_sq / h1 + s2_sq / h2) ** 2
        dominator = ((s1_sq / h1) ** 2) / (h1 - 1) + ((s2_sq / h2) ** 2) / (h2 - 1)
        df = numerator / dominator

        # t-statistic
        t_stat = (x1 - x2) / np.sqrt(s1_sq / h1 + s2_sq / h2)

        # Two-sided p-value
        p = 2 * (1 - t.cdf(abs(t_stat), df=df))

        return p
    
    if type == 'oneway':
        return one_way_t_test(dt1, mu)
    elif type == 'independent':
        return independent_t_test(dt1, dt2)
    elif type == 'paired':
        return paired_t_test(dt1, dt2)
    elif type == 'yuen':
        return yuan_welch_t_test(dt1, dt2, gamma=0.2)
    else:
        raise ValueError('Only oneway, independent, paired and yuen are valid type!')

# Mann-Whitney U Test
def mann_whitney_u_test(dt1: list, dt2: list):
    # Mann-Whitney
    from math import sqrt, inf
    from scipy.integrate import quad

    n1, n2 = len(dt1), len(dt2)
    combined = dt1 + dt2
    combined_ranks = get_rank(combined)

    r1 = sum(combined_ranks[:n1])
    r2 = sum(combined_ranks[n1:])

    u1 = n1 * n2 + ((n1 * (n1 + 1)) / 2) - r1
    u2 = n1 * n2 + ((n2 * (n2 + 1)) / 2) - r2
    u_stat = min(u1, u2)

    # Approximation by Z-distribution
    mu_u = (n1 * n2) / 2
    sigma_u = sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
    z_stat = (u_stat - mu_u) / sigma_u

    p_val_tail, _ = quad(t_pdf_log, abs(z_stat), inf, args = (999,)) # When set df = 999, t-distribution will be Z-distribution
    p_value = 2 * p_val_tail

    return {
        'u': u_stat,
        'p': p_value,
    }

# Wilcoxon Signed Rank Test
def wilcoxon_signed_rank_test(dt1: list, dt2: list):
    from math import sqrt, inf
    from scipy.integrate import quad

    # Calculate the difference and remove those equal to zero
    diffs = [a - b for a, b in zip(dt1, dt2)]
    filtered_diffs = [d for d in diffs if d != 0]
    n = len(filtered_diffs)

    # When two variables are indifferent
    if n == 0:
        return {
            'w': 0.0,
            'p': 1.0,
        }
    
    # Ranking the absolute value
    abs_diffs = [abs(d) for d in filtered_diffs]
    ranks = get_rank(abs_diffs)

    # Calculate the sum of ranks for both the plus and the minus
    w_plus = 0
    w_minus = 0
    for i in range(n):
        if filtered_diffs[i] > 0:
            w_plus += ranks[i]
        else:
            w_minus += ranks[i]
        
    w_stat = min(w_plus, w_minus)

    # Approximation by Z-distribution
    mu_w = n * (n + 1) / 4
    sigma_w = sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z_stat = (w_stat - mu_w) / sigma_w

    p_val_tail, _ = quad(t_pdf_log, abs(z_stat), inf, args = (999,))
    p_value = 2 * p_val_tail

    return {
        'w': w_stat,
        'p': p_value,
        'w_plus': w_plus,
        'w_minus': w_minus
    }

# Kruskal-Wallis Test
def kruskal_wallis_test(*groups):
    from math import inf
    from scipy.integrate import quad

    all_data = []
    group_info = [] # Length for each group
    for g in groups:
        all_data.extend(g)
        group_info.append(len(g))
    
    N = len(all_data)
    k = len(groups)

    all_ranks = get_rank(all_data)

    # Calculate sum of rank for each group
    rank_sums = []
    current_pos = 0
    for length in group_info:
        group_ranks = all_ranks[current_pos: current_pos + length]
        rank_sums.append(sum(group_ranks))
        current_pos += length
    
    # Calculate H-statisitc
    sum_sq_rank_div_n = 0
    for Ri, ni in zip(rank_sums, group_info):
        sum_sq_rank_div_n += (Ri**2) / ni
    
    h_stat = (12 / (N * (N + 1))) * sum_sq_rank_div_n - 3 * (N + 1)

    # Calculate p-value
    df = k - 1
    p_value, _ = quad(chi_square_pdf, h_stat, inf, args = (df,))

    return {
        'h': h_stat,
        'p': p_value,
        'df': df
    }

# Post Hoc Test (Bonferroni)
def post_hoc_bonferroni(*groups: list, method = 't_test'):
    from itertools import combinations

    k = len(groups)
    pairs = list(combinations(range(k), 2))
    m = len(pairs)
    alpha = 0.05
    adj_alpha = alpha / m

    results = []

    # Post hoc
    for i, j in pairs:
        g1, g2 = groups[i], groups[j]

        if method == 't_test':
            temp_res = t_test(g1, g2, type = 'independent')
            p_val = temp_res['p']
        elif method == 'mwu':
            temp_res = mann_whitney_u_test(g1, g2)
            p_val = temp_res['p']
        else:
            raise ValueError('Only t_test and mwu are valid type!')
        
        # Significance
        isSig = p_val < adj_alpha

        results.append(
            {
                'pair': (i+1, j+1),
                'p': p_val,
                'adj_alpha': adj_alpha,
                'is_sig': isSig
            }
        )
    
    return results

# Spearman's Rank Correlation Coefficient
def spearman_corr(x: list, y: list):
    # Get length info
    if len(x) != len(y):
        raise ValueError('Length of x and y must be equal!')

    # Ranking
    rank_x = get_rank(x)
    rank_y = get_rank(y)

    # Spearman
    rho, p_value = pearson_corr(rank_x, rank_y)

    return {
        'rho': rho,
        'p': p_value
    }

# Logistic Regression
def logistic_regression(X, y, lr = 0.01, iterations = 1500): # Where X is a matrix and y is a vector
    import numpy as np

    # Design matrix
    n_samples, n_features = X.shape
    X_design = np.hstack([np.ones((n_samples, 1)), X])
    weights = np.zeros(n_features + 1)

    # Sigmoid function
    def sigmoid(z: float):
        return 1 / (1 + np.exp(-z))
    
    # Gradient
    for _ in range(iterations):
        model = np.dot(X_design, weights)
        predictions = sigmoid(model)

        # Calculate gradient
        gradient = np.dot(X_design.T, (predictions - y)) / n_samples

        # Update weights
        weights -= lr * gradient

    return {
        'beta': weights,
        'pred_prob': lambda new_X: sigmoid(np.dot(np.hstack([np.ones((new_X.shape[0], 1)), new_X]), weights)),
        'pred': lambda new_X: (sigmoid(np.dot(np.hstack([np.ones((new_X.shape[0], 1)), new_X]), weights)) >= 0.5).astype(int)
    }