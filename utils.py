from sklearn.ensemble import RandomForestRegressor 
from scipy.stats import norm
import numpy as np
import pandas as pd

def overwrite_merge(df, merge_df, on, how='left'):
    df_merged = df.merge(merge_df, on=on, how=how, suffixes=('', '_new'))
    for col in merge_df.columns:
        if col in df_merged and col + '_new' in df_merged:
            df_merged[col] = df_merged[col + '_new']
            df_merged.drop(columns=[col + '_new'], inplace=True)
    return df_merged

def rejection_sample(df, target_col, accept_prob_col, n_samples, random_seed=None):
    accepted_rows = []
    while len(accepted_rows) < n_samples:
        sample = df.sample(n=n_samples, replace=True).reset_index(drop=True)
        u = np.random.uniform(0, 1, size=n_samples)
        accepted = sample[u < sample[accept_prob_col].values]
        accepted_rows.append(accepted[target_col])
    return pd.concat(accepted_rows, ignore_index=True).head(n_samples).reset_index(drop=True)

def PGAE_est_ci(PGAE_df, X, F, Y, alpha=0.95, K=5):
    """
    Calculate the confidence interval for the treatment effect using the PGAE method.
    
    Parameters:
    - PGAE_df: DataFrame containing the data with columns for features, treatment, and outcome.
    - X: List of feature column names.
    - F: Name of the prediction column.
    - Y: Name of the outcome column.
    - alpha: Significance level for the confidence interval.
    
    Returns:
    - tau_PGAE: Estimated treatment effect.
    - lower_bound: Lower bound of the confidence interval.
    - upper_bound: Upper bound of the confidence interval.
    """

    PGAE_labeled = PGAE_df[PGAE_df['true_label'] == 1]
    unlabeled_indices = PGAE_df[PGAE_df['true_label'] == 0].index.to_list()
    n = len(PGAE_labeled)
    N = len(unlabeled_indices)
    labeled_indices = PGAE_labeled.index.to_list()

    tau_PGAE = 0
    var_PGAE = 0
    for i in range(K):
        fold1 = labeled_indices[i*n//K: (i+1)*n//K]
        fold2 = labeled_indices[: i*n//K] + labeled_indices[(i+1)*n//K:]

        unlabeled_fold1 = unlabeled_indices[i*N//K: (i+1)*N//K]

        model_mu = RandomForestRegressor(n_estimators=20, random_state=42)
        model_mu.fit(PGAE_df.loc[fold2, X], PGAE_df.loc[fold2, Y])
        PGAE_df.loc[fold1+unlabeled_fold1, 'mu'] = model_mu.predict(PGAE_df.loc[fold1+unlabeled_fold1, X])

        model_tau = RandomForestRegressor(n_estimators=100, random_state=42)
        model_tau.fit(PGAE_df.loc[fold2, X+[F]], PGAE_df.loc[fold2, Y])
        PGAE_df.loc[fold1+unlabeled_fold1, 'tau'] = model_tau.predict(PGAE_df.loc[fold1+unlabeled_fold1, X+[F]])

    PGAE_df['psi'] = PGAE_df['true_pmf'] / PGAE_df['sample_pmf'] / PGAE_df['exp_prob'] * (PGAE_df['true_label'] * (PGAE_df[Y] - PGAE_df['tau']) - PGAE_df['exp_prob'] * (PGAE_df['mu'] - PGAE_df['tau']))

    # Aggregate the mean of true_pmf by group
    df_summary = PGAE_df.groupby(X).agg({'true_pmf': 'mean', 'mu': 'mean'}).reset_index()
    tau_PGAE = np.sum(df_summary['mu'].values * df_summary['true_pmf'].values) / np.sum(df_summary['true_pmf'].values)
    tau_PGAE += PGAE_df['psi'].mean()

    var_PGAE = PGAE_df['psi'].var()
    coef = norm.ppf(1 - (1 - alpha) / 2)
    tau_var = var_PGAE / len(PGAE_df)

    return tau_PGAE, tau_PGAE - np.sqrt(tau_var) * coef, tau_PGAE + np.sqrt(tau_var) * coef


# def PGAE_est_ci(PGAE_df, X, F, Y, alpha=0.95, K=5):
#     """
#     Calculate the confidence interval for the treatment effect using the PGAE method.
    
#     Parameters:
#     - PGAE_df: DataFrame containing the data with columns for features, treatment, and outcome.
#     - X: List of feature column names.
#     - F: Name of the prediction column.
#     - Y: Name of the outcome column.
#     - alpha: Significance level for the confidence interval.
    
#     Returns:
#     - tau_PGAE: Estimated treatment effect.
#     - lower_bound: Lower bound of the confidence interval.
#     - upper_bound: Upper bound of the confidence interval.
#     """

#     PGAE_labeled = PGAE_df[PGAE_df['true_label'] == 1]
#     unlabeled_indices = PGAE_df[PGAE_df['true_label'] == 0].index.to_list()
#     n = len(PGAE_labeled)
#     labeled_indices = PGAE_labeled.index.to_list()

#     tau_PGAE = 0
#     var_PGAE = 0
#     for i in range(K):
#         fold1 = labeled_indices[i*n//K: (i+1)*n//K]
#         fold2 = labeled_indices[: i*n//K] + labeled_indices[(i+1)*n//K:]

#         model_mu = RandomForestRegressor(n_estimators=100, random_state=42)
#         model_mu.fit(PGAE_df.loc[fold2, X], PGAE_df.loc[fold2, Y])
#         PGAE_df['mu'] = model_mu.predict(PGAE_df[X])

#         model_tau = RandomForestRegressor(n_estimators=100, random_state=42)
#         model_tau.fit(PGAE_df.loc[fold2, X+[F]], PGAE_df.loc[fold2, Y])
#         PGAE_df['tau'] = model_tau.predict(PGAE_df[X+[F]])

#         PGAE_df['psi'] = PGAE_df['true_pmf'] / PGAE_df['sample_pmf'] / PGAE_df['exp_prob'] * (PGAE_df['true_label'] * (PGAE_df[Y] - PGAE_df['tau']) - PGAE_df['exp_prob'] * (PGAE_df['mu'] - PGAE_df['tau']))

#         # Aggregate the mean of true_pmf by group
#         df_summary = PGAE_df.groupby(X).agg({'true_pmf': 'mean'}).reset_index()
#         predictions = model_mu.predict(df_summary[X])
#         tau = np.sum(predictions * df_summary['true_pmf'].values)
#         tau += PGAE_df.loc[fold1 + unlabeled_indices, 'psi'].mean()

#         tau_PGAE += tau * len(fold1) / len(PGAE_labeled)
#         var_PGAE += PGAE_df.loc[fold1 + unlabeled_indices, 'psi'].var() * len(fold1) / len(PGAE_labeled)

#     coef = norm.ppf(1 - (1 - alpha) / 2)
#     tau_var = var_PGAE / (len(PGAE_labeled)/K + len(unlabeled_indices))

#     return tau_PGAE, tau_PGAE - np.sqrt(tau_var) * coef, tau_PGAE + np.sqrt(tau_var) * coef


def adaptive_PGAE_est_ci(PGAE_df, X, F, Y, alpha=0.95, batch_size=100):

    PGAE_df['mu'] = PGAE_df[Y].astype(float)    
    PGAE_df['tau'] = PGAE_df[Y].astype(float)
    PGAE_df['psi'] = 0.0
    
    for i in range(batch_size, len(PGAE_df), batch_size):
        batch = PGAE_df.iloc[i:i+batch_size].index
        if len(batch) < batch_size:
            break
        training_data = PGAE_df.iloc[:i]
        model_mu = RandomForestRegressor(n_estimators=100, random_state=42)
        model_mu.fit(training_data[X], training_data[Y])
        PGAE_df.loc[batch, 'mu'] = model_mu.predict(PGAE_df.loc[batch, X])

        model_tau = RandomForestRegressor(n_estimators=100, random_state=42)
        model_tau.fit(training_data[X+[F]], training_data[Y])
        PGAE_df.loc[batch, 'tau'] = model_tau.predict(PGAE_df.loc[batch, X+[F]])

    PGAE_df['psi'] = PGAE_df['true_pmf'] / PGAE_df['sample_pmf'] / PGAE_df['exp_prob'] * (PGAE_df['true_label'] * (PGAE_df[Y] - PGAE_df['tau']) - PGAE_df['exp_prob'] * (PGAE_df['mu'] - PGAE_df['tau']))

    df_summary = PGAE_df.groupby(X).agg({'true_pmf': 'mean'}).reset_index()
    predictions = model_mu.predict(df_summary[X])
    tau_PGAE = np.sum(predictions * df_summary['true_pmf'].values) / np.sum(df_summary['true_pmf'].values)
    tau_PGAE += PGAE_df['psi'].mean()
    var_PGAE = PGAE_df['psi'].var()

    coef = norm.ppf(1 - (1 - alpha) / 2)
    tau_var = var_PGAE / (len(PGAE_df))
    lower_bound = tau_PGAE - np.sqrt(tau_var) * coef
    upper_bound = tau_PGAE + np.sqrt(tau_var) * coef
    return tau_PGAE, lower_bound, upper_bound
