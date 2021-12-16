import csv
import os
import shutil
import pandas as pd

path_e = '/bop2/bop23/Emotion'

label_txt_single = []  # [('S010', '002', '00000014', 7), (), ()]
# list_png = []  # [path, label]

merge_dir_a = '/bop2/bop23/cohn-kanade-images'
merge_dir_base = os.path.join(os.getcwd(), 'CK+')  # '/bop2/discriiminator_capsule/CK+'
# merge_dir_base = '/bop2/discriiminator_capsule/CK+'

path_csv = '/bop2/discriiminator_capsule/datasets'
path_current = os.getcwd()  # '/bop2/discriiminator_capsule'

path = os.getcwd() + '/datasets/'
data_all = os.path.join(path, 'data_all.csv')
data_a = os.path.join(path, 'cohn_corres.csv')
data_b = os.path.join(path, 'data_apex.csv')

output_file_name = data_a
data_test = os.path.join(path, 'cohn.csv')

columns = ['data', 'subject', 'clip', 'label', 'onset_frame', 'corresponding_frame',
           'offset_frame', 'onset_frame_path', 'corresponding_frame_path', 'offset_frame_path',
           'data_sub', 'domain_label']

percent_value = 0.4


# 遍历文件夹及其子文件夹中的文件，并存储在一个列表中

# 输入文件夹路径、空文件列表[]

# 返回 文件列表Filelist,包含文件名（完整路径）
# 保存所有的label
def get_filelist(dirs, fileslist, pathslist, cor_label, list_png):
    newDir = dirs  # 存放所有路径（txt+png）

    if os.path.isfile(dirs):

        pathslist.append(dirs)

        # 若只是要返回文件文，使用这个
        filename = os.path.basename(dirs)  # 路径中最末尾一个文件夹的名字
        fileslist.append(filename)

        if os.path.splitext(filename)[1] == '.txt':
            label = read_txt(dirs)
            print("{} 文件中的内容：{}".format(dirs, label))

            # 匹配
            path_split_out = path_split(dirs, label)  # ['S010', '002', '00000014', 7]
            merge_path_src = path_merge(merge_dir_a, path_split_out)  #
            merge_path_dst = path_merge(merge_dir_base, path_split_out)  # '/bop2/discriiminator_capsule/CK+/S010/002'
            list_index = -1
            for i in os.listdir(merge_path_src + '/'):
                if i == '.DS_Store':
                    os.remove(merge_path_src + '/.DS_Store')
                    break

            shutil.copytree(merge_path_src, merge_path_dst, True)

            cor_label.append(label)
            list_png.append((merge_path_dst, label))
            # write_to_csv(path_csv, )
            pass

    elif os.path.isdir(dirs):

        for s in os.listdir(dirs):
            # 如果需要忽略某些文件夹，使用以下代码

            # if s == "xxx":

            # continue

            newDir = os.path.join(dirs, s)

            get_filelist(newDir, fileslist, pathslist, cor_label, list_png)

    return fileslist, pathslist, cor_label, list_png


# 读取txt文件中的内容
def read_txt(txt_file):
    with open(txt_file, 'r') as f:
        line = f.readline()
    return int(float(line.strip()))


# 拆开路径
def path_split(path, label):  # '/bop2/bop23/B/S004/002/S010_002_00000014_emotion.txt'
    split_txt = os.path.basename(path)  # ('S010_002_00000014_emotion.txt')
    split_txt = os.path.splitext(split_txt)  # ('S010_002_00000014_emotion', '.txt')
    split_txt = split_txt[0]
    split_txt = split_txt.split('_')  # ['S010', '002', '00000014', 'emotion']
    split_txt[3] = label
    # for i in split_txt:
    #     txt_single.append(i)  # ['S010', '002', '00000014', 7]

    return split_txt  # ['S010', '002', '00000014', 7]


# 合成路经
def path_merge(merge_d, list_m):  # ['S010', '002', '00000014', 7]
    # merge_dir = '/bop2/bop23/B'
    merge_d = os.path.join(merge_d, list_m[0])
    merge_d = os.path.join(merge_d, list_m[1])
    return merge_d  # '/bop2/bop23/B/S010/002'


# 写入csv文件
def write_to_csv(dst_fname, dst_dict):
    # dataframe = pd.DataFrame().append(dst_dict, ignore_index=True)
    # dataframe.to_csv(dst_fname, mode='a', header=False, index=False, sep=',')
    data = pd.DataFrame(dst_dict, index=[0])
    data.to_csv(dst_fname, index=False, mode='a', header=False, columns=dst_dict.keys())


def save_to_csv(path, l_png):
    png_list = []
    data = 'CK+'
    domain_label = 0
    for home, dirs, files in os.walk(path):  # home= '/bop2/discriiminator_capsule/CK+/S133/010'
        #  dirs= ['S133', 'S130', 'S505', 'S155', 'S074', 'S081', 'S076', 'S160', 'S035',,,]
        #  files=['S133_010_00000010.png', 'S133_010_00000003.png', 'S133_010_00000007.png', ,,]
        count = 0
        for filename in files:
            # 文件名列表，包含完整路径
            files.sort()
            png_list.append(os.path.join(home, filename))
            count += 1

            if count == len(files):
                print("home:{}".format(home))
                # # 文件名列表，只包含文件名

                # Filelist.append(filename)
                home_s = home.split('/')
                subject = home_s[-2]
                clip = home_s[-1]

                # print([pn for pn, (i, j) in enumerate(l_png) if i == home])
                for pn, (i, j) in enumerate(l_png):
                    if i == home:
                        # print("pn:{} i:{} j:{}  type(j):{}".format(pn, i, j, type(j)))
                        label = j
                        break
                # files.sort()
                loacte_index = int(len(files) * percent_value)
                onset_frame = files[0].split('.')[0]
                onset_frame = onset_frame.split('_')[-1]
                corresponding_frame1 = files[loacte_index - 1].split('.')[0]
                corresponding_frame1 = corresponding_frame1.split('_')[-1]
                corresponding_frame2 = files[loacte_index].split('.')[0]
                corresponding_frame2 = corresponding_frame2.split('_')[-1]
                corresponding_frame3 = files[loacte_index + 1].split('.')[0]
                corresponding_frame3 = corresponding_frame3.split('_')[-1]
                offset_frame = files[-1].split('.')[0]
                offset_frame = offset_frame.split('_')[-1]
                onset_frame_path = data + "/" + subject + "/" + clip + "/" + files[0]
                corresponding_frame1_path = data + "/" + subject + "/" + clip + "/" + files[loacte_index - 1]
                corresponding_frame2_path = data + "/" + subject + "/" + clip + "/" + files[loacte_index]
                corresponding_frame3_path = data + "/" + subject + "/" + clip + "/" + files[loacte_index + 1]
                offset_frame_path = data + "/" + subject + "/" + clip + "/" + files[-1]
                data_sub = data + '_' + subject

                # list_add = [data, subject, clip, label, onset_frame, corresponding_frame,
                #             offset_frame, onset_frame_path, corresponding_frame_path,
                #             offset_frame_path]

                dict_add1 = {'data': data, "subject": subject, 'clip': clip, "label": label,
                             'onset_frame': onset_frame, "corresponding_frame": corresponding_frame1,
                             'offset_frame': offset_frame, 'onset_frame_path': onset_frame_path,
                             "corresponding_frame_path": corresponding_frame1_path,
                             'offset_frame_path': offset_frame_path, 'data_sub': data_sub,
                             'domain_label': domain_label}

                dict_add2 = {'data': data, "subject": subject, 'clip': clip, "label": label,
                             'onset_frame': onset_frame, "corresponding_frame": corresponding_frame2,
                             'offset_frame': offset_frame, 'onset_frame_path': onset_frame_path,
                             "corresponding_frame_path": corresponding_frame2_path,
                             'offset_frame_path': offset_frame_path, 'data_sub': data_sub,
                             'domain_label': domain_label}
                dict_add3 = {'data': data, "subject": subject, 'clip': clip, "label": label,
                             'onset_frame': onset_frame, "corresponding_frame": corresponding_frame3,
                             'offset_frame': offset_frame, 'onset_frame_path': onset_frame_path,
                             "corresponding_frame_path": corresponding_frame3_path,
                             'offset_frame_path': offset_frame_path, 'data_sub': data_sub,
                             'domain_label': domain_label}

                for dic in dict_add1, dict_add2, dict_add3:
                    write_to_csv(output_file_name, dic)

    return len(png_list)


def csv_merge(dst, src1, src2):
    # # 向微表情中添加域标签=1
    # data = pd.read_csv(data_b)
    # data['domain_label'] = 1
    # data.to_csv(data_b, index=False)

    # 合并表情和微表情
    db = pd.read_csv(data_b)
    db.to_csv(data_all, encoding='utf-8', mode='w', index=False)
    da = pd.read_csv(data_a)
    da = da[(da['label'].values == 5) | (da['label'].values == 6) | (da['label'].values == 7)]
    # da = da[(da['label'].values == 6) | (da['label'].values == 7)]
    da.loc[da.label == 6, 'label'] = 0
    da.loc[da.label == 5, 'label'] = 1
    da.loc[da.label == 7, 'label'] = 2
    da.to_csv(data_all, encoding='utf-8', mode='a', index=False, header=False)
    da.rename(columns={'corresponding_frame': 'apex_frame', 'corresponding_frame_path': 'apex_frame_path'},
              inplace=True)
    da.to_csv(data_test, encoding='utf-8', mode='w', index=False)


if __name__ == '__main__':
    files_list, paths_list, cor_labels, lists_png = get_filelist(path_e, [], [], [], [])

    print("文件数：{a}  路径数：{b}".format(a=(len(files_list)), b=len(paths_list)))

    for lt in zip(files_list, paths_list, cor_labels):
        print(lt)

    print("开始打印所有png：")
    lists_png = sorted(lists_png)
    for i in lists_png:
        print(i)

    with open(output_file_name, 'w', encoding="utf-8", newline="") as f:
        # key_data = headers
        # value_data = [line for line in rows]
        csv_writer = csv.writer(f)
        csv_writer.writerow(columns)

    # pd.DataFrame().to_csv(output_file_name, encoding='utf-8', index=False, columns=columns)
    files_l = save_to_csv(merge_dir_base, lists_png)
    csv_merge(data_all, data_a, data_b)

# /
#
# if __name__ == "__main__":
#
#     Filelist = get_filelist(dir)
#
#     print(len(Filelist))
#
#     for file in Filelist:
#         print(file)
