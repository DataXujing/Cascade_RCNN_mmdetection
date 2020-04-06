
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv

from mmdet.core import bingzao_classes

from glob import glob
from tqdm import tqdm
from PIL import Image
label_ids = {name: i + 1 for i, name in enumerate(bingzao_classes())}

print(label_ids)

# label_ids = {"Barrett":1,"CX":2,"FLXSGY":3,"HJQ":4,"JCJZQA":5,"JCXR":6,"JCZA":7,"JZQWA":8,
# "JS":9,"KYXJCY":10,"MXWSXWY":11,"QP":12,"QG":13,"QTMH":14,"QTQPGY":15,"SGJMQZ":16,"SGZA":17,
# "TW":18,"WKY":19,"WZA":20,"YD":21,"ZZ":22}


def label_rename(label_str):
    if ("barrett食管" in label_str) or ("barrett 食管" in label_str) or ("barrett" in label_str) or ("Barrett食管" in label_str):
        return 'Barrett'
    elif "出血" in label_str:
        return "CX"
    elif "反流性食管炎" in label_str:
        return 'FLXSGY'
    elif "活检钳" in label_str:
        return "HJQ"
    elif "结肠进展期癌" in label_str:
        return 'JCJZQA'
    elif ("结肠息肉" in label_str) or ("息肉" in label_str):
        return 'JCXR'
    elif ("结肠早癌" in label_str) or ("早期结直肠癌" in label_str):
        return 'JCZA'
    elif "进展期胃癌" in label_str:
        return 'JZQWA'
    elif "镜身" in label_str:
        return 'JS'
    elif "溃疡性结肠炎" in label_str:
        return 'KYXJCY'
    elif "慢性萎缩性胃炎" in label_str:
        return 'MXWSXWY'      
    elif "气泡" in label_str:
        return 'QP'  
    elif ("强光" in label_str) or ("反光" in label_str):
        return 'QG'  
    elif "全图模糊" in label_str:
        return 'QTMH'  
    elif "全图气泡光晕" in label_str:
        return 'QTQPGY' 
    elif "食管静脉曲张" in label_str:
        return 'SGJMQZ'       
    elif "食管早癌" in label_str:
        return 'SGZA' 
    elif "体外" in label_str:
        return 'TW'
    elif ("胃溃疡" in label_str) or ("胃良性溃疡" in label_str) or ("胃恶性溃疡" in label_str) or ("溃疡" in label_str):
        return 'WKY'
    elif ("胃早癌" in label_str) or ("早期胃癌" in label_str):
        return 'WZA'
    elif "牙垫" in label_str:
        return 'YD'
    elif "褶皱" in label_str:
        return 'ZZ'


def get_segmentation(points):

    return [points[0], points[1], points[2] + points[0], points[1],
             points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]


def parse_xml(xml_path, img_id, anno_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation = []
    for obj in root.findall('object'):
        name_text = obj.find('name').text
        name = label_rename(name_text)
        if name not in ["Barrett","CX","FLXSGY","HJQ","JCJZQA","JCXR","JCZA","JZQWA","JS","KYXJCY","MXWSXWY","QP","QG","QTMH","QTQPGY","SGJMQZ","SGZA","TW","WKY","WZA","YD","ZZ"]:
            print(xml_path)
            continue
        category_id = label_ids[name]
        bnd_box = obj.find('bndbox')
        xmin = int(bnd_box.find('xmin').text)
        ymin = int(bnd_box.find('ymin').text)
        xmax = int(bnd_box.find('xmax').text)
        ymax = int(bnd_box.find('ymax').text)
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        area = w*h
        segmentation = get_segmentation([xmin, ymin, w, h])
        annotation.append({
                        "segmentation": segmentation,
                        "area": area,
                        "iscrowd": 0,
                        "image_id": img_id,
                        "bbox": [xmin, ymin, w, h],
                        "category_id": category_id,
                        "id": anno_id,
                        "ignore": 0})
        anno_id += 1
    return annotation, anno_id


def cvt_annotations(img_path, xml_path, out_file):
    images = []
    annotations = []

    # xml_paths = glob(xml_path + '/*.xml')
    img_id = 1
    anno_id = 1
    for img_path in tqdm(glob(img_path + '/*.jpg')):
        w, h = Image.open(img_path).size
        img_name = osp.basename(img_path)
        img = {"file_name": img_name, "height": int(h), "width": int(w), "id": img_id}
        images.append(img)

        xml_file_name = img_name.split('.')[0] + '.xml'
        xml_file_path = osp.join(xml_path, xml_file_name)
        annos, anno_id = parse_xml(xml_file_path, img_id, anno_id)
        annotations.extend(annos)
        img_id += 1

    categories = []
    for k,v in label_ids.items():
        categories.append({"name": k, "id": v})
    final_result = {"images": images, "annotations": annotations, "categories": categories}
    mmcv.dump(final_result, out_file)
    return annotations


def main():
    # train
    xml_path = 'data/coco/train/annotations'
    img_path = 'data/coco/train/JPEGImages'
    print('processing {} ...'.format("xml format annotations"))
    cvt_annotations(img_path, xml_path, 'data/coco/annotations/train.json')
    print('Train Done!')


    # test
    xml_path = 'data/coco/test/annotations'
    img_path = 'data/coco/test/JPEGImages'
    print('processing {} ...'.format("xml format annotations"))
    cvt_annotations(img_path, xml_path, 'data/coco/annotations/test.json')
    print('Test Done!')


if __name__ == '__main__':
    main()
