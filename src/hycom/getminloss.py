import os
import cudf

lines = []
log_dict = {'max_depth': [], 'min_child_weight': [], 'eta': [], 'subsample': [], 'boost_rounds': [], 'mean_error': []}

with open('logs.txt', 'r') as logs:
    for num, i in enumerate(logs):
        if num == 0:
            continue
        i = i.replace('\n', '')
        max_depth,min_child_weight,eta,subsample,boost_rounds,mean_error = i.split(',')
        log_dict['max_depth'].append(max_depth) 
        log_dict['min_child_weight'].append(min_child_weight)
        log_dict['eta'].append(eta) 
        log_dict['subsample'].append(subsample)
        log_dict['boost_rounds'].append(boost_rounds) 
        log_dict['mean_error'].append(mean_error)


log = cudf.DataFrame(log_dict)
dtypes = {'max_depth': 'float32', 'min_child_weight': 'float32', 'eta': 'float32', 'subsample': 'float32', 'boost_rounds': 'float32', 'mean_error': 'float32'}
log = log.astype(dtypes)
log = log.sort_values(['mean_error', 'max_depth', 'eta'], ascending=[True, True, True], ignore_index=True)
print(log.dtypes)

filter = log['eta'] <= 0.01
log_filter = log.loc[filter]

print(log_filter.iloc[:15, :])
log.to_csv('logs.csv', ignore_index=True)