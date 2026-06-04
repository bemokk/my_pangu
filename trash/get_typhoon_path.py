import requests
import json
import re
import time
import pandas as pd
import datetime


def date_pred(date, deltahour):
    """
    date: yyyymmddHHMM, string
    deltahour: hours, integer
    """
    time = datetime.datetime.strptime(date, "%Y%m%d%H%M")
    new_date = (time + datetime.timedelta(hours=deltahour)).strftime("%Y%m%d%H%M")
    return new_date


def get_type(date_type):
    item = {'TC': '热带气旋', 'TD': '热带低压', 'TS': '热带风暴', 'STS': '强热带风暴',
            'TY': '台风', 'STY': '强台风', 'SuperTY': '超强台风', '': '', }
    return item.get(date_type, '')


def get_tc_info_by_id(tc_id, tc_num, name_cn, name_en):
    """
    直接通过台风ID获取台风路径信息
    tc_id: 台风ID，如3064324
    tc_num: 台风编号，如2509
    name_cn: 中文名
    name_en: 英文名
    """
    t = int(round(time.time() * 1000))  # 13位时间戳
    url = f'http://typhoon.nmc.cn/weatherservice/typhoon/jsons/view_{tc_id}?t={t}&callback=typhoon_jsons_view_{tc_id}'

    try:
        html_obj = requests.get(url, headers=headers, verify=False, timeout=10).text
        data = json.loads(re.match(".*?({.*}).*", html_obj, re.S).group(1))['typhoon']
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None

    # 建立字典
    info_dicts = {'tc_num': tc_num,  # 编号
                  'name_cn': name_cn,  # 中文名
                  'name_en': name_en,  # 英文名
                  'dateUTC': [],  # 日期 UTC
                  'dateCST': [],  # 日期 CST
                  'vmax': [],  # 最大风速 m/s
                  'grade': [],  # 等级
                  'latTC': [],  # 位置deg
                  'lonTC': [],
                  'mslp': [],  # 中心气压hPa
                  'attr': []}  # 属性,预报forecast，实况analysis

    # 先遍历实况
    for v in data[8]:
        info_dicts['dateUTC'].append(v[1])
        info_dicts['dateCST'].append(date_pred(v[1], 8))  # UTC to CST
        info_dicts['vmax'].append(v[7])
        info_dicts['grade'].append(get_type(v[3]))
        info_dicts['lonTC'].append(v[4])
        info_dicts['latTC'].append(v[5])
        info_dicts['mslp'].append(v[6])
        info_dicts['attr'].append('analysis')

    # 最新预报时刻
    dateUTC0 = info_dicts['dateUTC'][-1] if info_dicts['dateUTC'] else None

    # 如果有预报数据，则添加最新预报
    if dateUTC0 and data[8][-1][11] and 'BABJ' in data[8][-1][11]:
        BABJ_list = data[8][-1][11]['BABJ']
        for i in range(len(BABJ_list)):
            pred_hour = int(BABJ_list[i][0])  # 预报时效，hour
            dateUTC_pred = date_pred(dateUTC0, pred_hour)
            info_dicts['dateUTC'].append(dateUTC_pred)
            info_dicts['dateCST'].append(date_pred(dateUTC_pred, 8))
            info_dicts['vmax'].append(BABJ_list[i][5])
            info_dicts['grade'].append(get_type(BABJ_list[i][7]))
            info_dicts['lonTC'].append(BABJ_list[i][2])
            info_dicts['latTC'].append(BABJ_list[i][3])
            info_dicts['mslp'].append(BABJ_list[i][4])
            info_dicts['attr'].append('forecast')

    tc_info = pd.DataFrame(info_dicts)
    return tc_info


if __name__ == "__main__":
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # 第2509号台风"罗莎"(KROSA)的信息
    tc_id = "3062724"  # 从URL https://typhoon.nmc.cn/web.html?tid=3064324 中获取的tid httpstyphoon.nmc.cnweb.htmltid=3062724
    tc_num = "2507"  # 台风编号
    name_cn = "范斯高"  # 中文名
    name_en = "Francisco"  # 英文名

    print(f"正在获取第{tc_num}号台风'{name_cn}'({name_en})的路径信息...")

    data = get_tc_info_by_id(tc_id, tc_num, name_cn, name_en)

    if data is not None:
        print(f"第{tc_num}号台风'{name_cn}'({name_en})路径信息:")
        print(data)

        # 保存到CSV文件
        filename = f"typhoon_{tc_num}_{name_en}.csv"
        data.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"数据已保存到: {filename}")

        # 显示基本信息
        print("\n台风基本信息:")
        print(f"台风编号: {tc_num}")
        print(f"中文名: {name_cn}")
        print(f"英文名: {name_en}")
        print(f"总记录数: {len(data)}条")
        print(f"实况记录数: {len(data[data['attr'] == 'analysis'])}条")
        print(f"预报记录数: {len(data[data['attr'] == 'forecast'])}条")
    else:
        print("获取台风信息失败")