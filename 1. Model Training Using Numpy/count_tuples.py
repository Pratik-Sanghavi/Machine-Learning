def count_tuples(tuples_list):
    res = {}
    for tuple in tuples_list:
        if tuple in res.keys():
            res[tuple] +=1
        else:
            res[tuple] = 1
    return res

tuples_list = [(1,2,3), (1,2), (6,7,8), (1,2,3), (3,4), (1,2,3), (6,7,8)]

print(count_tuples(tuples_list))