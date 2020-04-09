import glob
import pandas as pd
import yaml


def generate_defs(config):
    """
    This scripts reads all the txt file from a temp location and trasfrorm and saves the
    data as a csv with `term` and `definition`.
    """
    txt_files = glob.glob(config['path']['temp_path'])
    list_defs = []
    for txt in txt_files:
        with open(txt, 'r', errors='ignore') as f:
            def_data = f.readlines()
            temp_list = []
            for item in def_data:
                line = item.replace('\n', '').replace('\t', 'Alias:').split(":")
                if 'CUI' in line:
                    temp_dict = {}
                    name = line[-1].replace('\n', '')
                    cui = line[1].split(",")[0]
                    temp_dict['Term'] = name
                # 		        temp_dict['CUI'] = cui
                # 		    if 'TUI(s)' in line:
                # 		        temp_dict['TUI'] = line[1]
                if 'Definition' in line:
                    definition = line[1]
                    temp_dict['Definition'] = definition
                # 		    if 'Alias' in line[0]:
                # 		        alias = line[-1]
                # 		        temp_dict['Alias'] = alias
                if temp_dict:
                    temp_list.append(temp_dict)

            df = pd.DataFrame(temp_list).drop_duplicates(subset="Term", keep='last')
            list_defs.append(df)

        # tt =   txt.split('.')[0] + '.csv'
        # df.to_csv(tt,index=False)
    frame = pd.concat(list_defs, axis=0, ignore_index=True)
    frame.drop_duplicates(inplace=True)
    frame = frame[['Term', 'Definition']]  # 'Alias','CUI','TUI']]
    frame.to_csv(config['path']['definition_file'], index=False)


if __name__ == "__main__":
    with open('.\config\config_defs.yaml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    generate_defs(config)