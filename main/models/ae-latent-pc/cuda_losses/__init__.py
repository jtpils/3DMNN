try:
    from tf_nndistance import nn_distance
    from tf_approxmatch import approx_match, match_cost
except:
    from tf_approxmatch import approx_match, match_cost
    from tf_nndistance import nn_distance
    print('External Losses (Chamfer-EMD) were not loaded.')
