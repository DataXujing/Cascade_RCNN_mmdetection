import os
import shutil
import uuid

def img_rename(img_path,anno_path,mode="train"):

    '''
    src: image path
    '''
    img_files = os.listdir(img_path)
    anno_files = os.listdir(anno_path)

    count = 0
    for img_file in img_files:
        file_name = img_file.rstrip("jpg")
        if file_name+"xml" in anno_files:
            new_file = str(uuid.uuid1()).replace("-","")
            src_img = os.path.join(img_path,img_file)
            dst_img = os.path.join("./data/coco/{}/JPEGImages".format(mode),new_file+".jpg")
            src_anno = os.path.join(anno_path,file_name+"xml")
            dst_anno = os.path.join("./data/coco/{}/annotations".format(mode),new_file+".xml")
            os.rename(src_img,dst_img)
            os.rename(src_anno,dst_anno)
            count += 1
        else:
            print(img_file)
            os.remove(os.path.join(img_path,img_file))

    print("[info] Rename Done: {} images".format(count))


if __name__ == "__main__":
    img_rename("./data/source/train/JPEGImages","./data/source/train/annotations",mode="train")
    img_rename("./data/source/test/JPEGImages","./data/source/test/annotations",mode="test")


