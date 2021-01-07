import os


def delete_spec_file_subname(path, str):
    dic_lis = [i for i in os.listdir(path)]
    print(dic_lis)
    dic_lis = [i for i in dic_lis if str in i]
    print(dic_lis)
    for file in dic_lis:
        os.remove(path+file)
        print(f'Delete file {file}')
