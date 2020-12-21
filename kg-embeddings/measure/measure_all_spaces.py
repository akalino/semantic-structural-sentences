import pandas as pd
import os

from statsmodels.stats.diagnostic import lilliefors

from measure_embedding_space import load_spaces, \
    compute_metrics, compute_variances, plot_variances, \
    compute_pca_frequency_degree, compute_isotropy


if __name__ == "__main__":
    """
    https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.lilliefors.html
    A variant of the test can be used to test the null hypothesis that data come
    from an exponentially distributed population, when the null hypothesis does
    not specify which exponential distribution.
    """
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir)
    dp = os.path.join(wd, "data", "RESIDE_KG/")
    models = ['transe', 'transh', 'transr', 'transd',
              'simple', 'complex', 'rescal'] # 'rotate' 'tucker']
    dims = [50, 100, 300]
    dset = 'riedel'
    kolo_smir_ent = []
    kolo_smir_rel = []
    pvals_ent = []
    pvals_rel = []
    check_dim = []
    check_mod = []
    iso_ent = []
    iso_rel = []
    for m in models:
        for d in dims:
            print('Model {} for dimension {}'.format(m, d))
            small_i = 10
            entities, relations = load_spaces(d, m, dset)
            ent_norm, ent_mean, rel_norm, rel_mean = compute_metrics(d, entities, relations)
            print('Entities norm {} and mean {}'.format(ent_norm, ent_mean))
            print('Relations norm {} and mean {}'.format(rel_norm, rel_mean))
            ent_var, rel_var = compute_variances(d, entities, relations)
            plot_variances(ent_var, 'Entity', m, d)
            plot_variances(rel_var, 'Relation', m, d)
            compute_pca_frequency_degree(d, entities, relations, dp, m)
            max_ent_iso, min_ent_iso, ent_iso_rat = compute_isotropy(entities)
            max_rel_iso, min_rel_iso, rel_iso_rat = compute_isotropy(relations)
            iso_ent.append(ent_iso_rat)
            iso_rel.append(rel_iso_rat)
            print('Entity space isotropy: {}'.format(ent_iso_rat))
            print('Relation space isotropy: {}'.format(rel_iso_rat))
            ks_ent, pv_ent = lilliefors(ent_var[0:small_i], dist='exp', pvalmethod='table')
            ks_rel, pv_rel = lilliefors(rel_var[0:small_i], dist='exp', pvalmethod='table')
            kolo_smir_ent.append(ks_ent)
            kolo_smir_rel.append(ks_rel)
            pvals_ent.append(pv_ent)
            pvals_rel.append(pv_rel)
            check_dim.append(d)
            check_mod.append(m)
            print('Lilliefors exponential: entities {}, relations {}'.format(pv_ent, pv_rel))
            if pv_ent < 0.05:
                print('Null rejected, entity variance is not exponential')
            else:
                print("Can't reject null, entity variance may be exponential")
            if pv_rel < 0.05:
                print('Null rejected, relation variance is not exponential')
            else:
                print("Can't reject null, relation variance may be exponential")
    res_df = pd.DataFrame({'model': check_mod,
                           'dim': check_dim,
                           'ks_ent': kolo_smir_ent,
                           'ks_rel': kolo_smir_rel,
                           'pv_ent': pvals_ent,
                           'pv_rel': pvals_rel,
                           'iso_ent': iso_ent,
                           'iso_rel': iso_rel})
    with open('kgemb-comp.tex', 'w') as tf:
        tf.write(res_df.to_latex(index=False))