def dump_dict_with_type(obj, name: str = ''):
    print(name + ': ', end='')
    if type(obj) == dict:
        print('dict')
        for k, v in obj.items():
            fullname = name + '.' + k
            dump_dict_with_type(v, fullname)
    elif type(obj) == list:
        print('list')
        for i, v in enumerate(obj):
            fullname = name + '.' + str(i)
            dump_dict_with_type(v, fullname)
    else:
        print(repr(obj), end='')
        print(" (%s)" % type(obj))
