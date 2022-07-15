import numpy as np
import statsmodels.api as sm

def ArithmeticOptimization(df, dep_atts, num_reps, num_vars):
    ncol = len(df.columns) - len(dep_atts)
    # print(dep_atts)

    indep_atts = df.drop(columns=dep_atts)

    for i in range(0, ncol - 1):
        for j in range(0, ncol - 1):
            # Using i+j instead of just j as an iterator avoids duplicates like x+y and y+x
            if i + j < ncol:
                indep_atts[indep_atts.columns[i] + '+' + indep_atts.columns[i + j]] = indep_atts.iloc[:,
                                                                                      i] + indep_atts.iloc[:, i + j]
                indep_atts[indep_atts.columns[i] + '*' + indep_atts.columns[i + j]] = indep_atts.iloc[:,
                                                                                      i] * indep_atts.iloc[:, i + j]

            # Avoids subtracting and dividing a number by itself
            if i != j:
                indep_atts[indep_atts.columns[i] + '-' + indep_atts.columns[j]] = indep_atts.iloc[:,
                                                                                  i] - indep_atts.iloc[:, j]
                indep_atts[indep_atts.columns[j] + '-' + indep_atts.columns[i]] = indep_atts.iloc[:,
                                                                                  j] - indep_atts.iloc[:, i]
                indep_atts[indep_atts.columns[i] + '/' + indep_atts.columns[j]] = indep_atts.iloc[:,
                                                                                  i] / indep_atts.iloc[:, j]
                indep_atts[indep_atts.columns[j] + '/' + indep_atts.columns[i]] = indep_atts.iloc[:,
                                                                                  j] / indep_atts.iloc[:, i]

    # print(indep_atts)

    # Replaces 'NaN's and 'Inf's with 0
    indep_atts.fillna(0, inplace=True)
    indep_atts.replace([np.inf, -np.inf], 0, inplace=True)

    for column in indep_atts.columns:
        indep_atts[column] = (indep_atts[column] - indep_atts[column].min()) / (
                    indep_atts[column].max() - indep_atts[column].min())

    if num_reps > 1:
        for x in range(1, num_reps):
            ncol = len(indep_atts.columns)

            for i in range(0, ncol - 1):
                for j in range(0, ncol - 1):
                    # Using i+j instead of just j as an iterator avoids duplicates like x+y and y+x
                    if i + j < ncol:
                        indep_atts[indep_atts.columns[i] + '+' + indep_atts.columns[i + j]] = indep_atts.iloc[:,
                                                                                              i] + indep_atts.iloc[:,
                                                                                                   i + j]
                        indep_atts[indep_atts.columns[i] + '*' + indep_atts.columns[i + j]] = indep_atts.iloc[:,
                                                                                              i] * indep_atts.iloc[:,
                                                                                                   i + j]

                    # Avoids subtracting and dividing a number by itself
                    if i != j:
                        indep_atts[indep_atts.columns[i] + '-' + indep_atts.columns[j]] = indep_atts.iloc[:,
                                                                                          i] - indep_atts.iloc[:, j]
                        indep_atts[indep_atts.columns[j] + '-' + indep_atts.columns[i]] = indep_atts.iloc[:,
                                                                                          j] - indep_atts.iloc[:, i]
                        indep_atts[indep_atts.columns[i] + '/' + indep_atts.columns[j]] = indep_atts.iloc[:,
                                                                                          i] / indep_atts.iloc[:, j]
                        indep_atts[indep_atts.columns[j] + '/' + indep_atts.columns[i]] = indep_atts.iloc[:,
                                                                                          j] / indep_atts.iloc[:, i]

            # Replaces 'NaN's and 'Inf's with 0
            indep_atts.fillna(0, inplace=True)
            indep_atts.replace([np.inf, -np.inf], 0, inplace=True)

            for column in indep_atts.columns:
                indep_atts[column] = (indep_atts[column] - indep_atts[column].min()) / (
                            indep_atts[column].max() - indep_atts[column].min())

                # Replaces 'NaN's and 'Inf's with 0
    indep_atts.fillna(0, inplace=True)
    indep_atts.replace([np.inf, -np.inf], 0, inplace=True)

    # print(indep_atts)

    reg_param = sm.OLS(df['Type_0'], indep_atts)
    reg_test = reg_param.fit()

    new_vars = reg_test.pvalues.nsmallest(n=num_vars).index

    return indep_atts[new_vars]