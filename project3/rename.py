import os
import re
import sys

def change_mark(path):
    old_names= os.listdir(path)
    i=0
    print((len(old_names)))
    for old_name in old_names:
        try:
            new_name=path+"\\image"+str(i)+".jpg"
            old_name=path+"\\"+old_name
            os.rename(old_name, new_name)
            i=i+1
        except Exception as e:
            print(e)


if __name__ == "__main__":
    clazz = ['accordion', 'airplanes', 'bass', 'bonsai', 'brain', 'buddha',
             'butterfly', 'camera', 'car_side', 'cellphone', 'chair', 'chandelier',
             'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head',
             'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar',
             'elephant', 'emu', 'euphonium', 'ewer', 'Faces', 'ferry', 'flamingo',
             'gramophone', 'grand_piano', 'hawksbill', 'hedgehog', 'helicopter', 'ibis',
             'joshua_tree', 'kangaroo', 'ketch', 'lamp']
    for i in range(len(clazz)):
        path = ".\\imgTest\\" + clazz[i]
        change_mark(path)
