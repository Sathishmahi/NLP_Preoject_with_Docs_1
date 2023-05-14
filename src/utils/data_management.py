from tqdm import tqdm
import random
import xml.etree.ElementTree as et
import re
# from src.stage_01_preparation_data import logging

def process_posts(fd_in,fd_out_train,fd_out_test,target_tag,split,column_names:list):
    line_no=1
    fd_out_test.write(f"{column_names[0]}\t{column_names[1]}\t{column_names[2]}\n")
    fd_out_train.write(f"{column_names[0]}\t{column_names[1]}\t{column_names[2]}\n")
    for line_info in tqdm(fd_in):
        try:
            fd_write= fd_out_train if random.random()>split else fd_out_test
            attr=et.fromstring(line_info).attrib
            tid=attr.get("Id","")
            label= 1 if target_tag in attr.get("Tags","") else 0
            title=re.sub("\s+", "", attr.get("Title","")).strip()
            body=re.sub("\s+", "", attr.get("Body","")).strip()
            text=f"{title} {body}"
            fd_write.write(f"{tid}\t{text}\t{label}\n")
            line_no+=1
        except Exception as e:
            mag=f"skip line no {line_no}  :  {e}\n"
            # logging.exception(msg=msg)


