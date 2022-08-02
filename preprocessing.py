from glob2 import glob
import os, shutil
from tqdm import tqdm


#------------------------------------------------------------------------------------
### Seperate file
def seperate_file():
	print('[*] Start Seperate Files')
	os.mkdir('images') if not os.path.exists('images') else None
	os.mkdir('./images/train') if not os.path.exists('./images/train') else None
	os.mkdir('./images/test') if not os.path.exists('./images/test') else None
	os.mkdir('./images/val') if not os.path.exists('./images/val') else None
	os.mkdir('labels') if not os.path.exists('labels') else None
	os.mkdir('./labels/train') if not os.path.exists('./labels/train') else None
	os.mkdir('./labels/test') if not os.path.exists('./labels/test') else None
	os.mkdir('./labels/val') if not os.path.exists('./labels/val') else None


	img_list = glob('./ship_dataset_v0/**.jpg')
	txt_list = glob('./ship_dataset_v0/**.txt')
	rang = [int(len(img_list)*9/10), int(len(img_list)*1/20)]



	for n, i in tqdm(enumerate(img_list)):
		basename = os.path.basename(i)
		shutil.move(i, './images/train/' + basename) if n < len(img_list[:rang[0]]) else None
		shutil.move(i, './images/val/' + basename) if n >= len(img_list[:rang[0]]) and n < len(img_list[:rang[0]+rang[1]]) else None
		shutil.move(i, './images/test/' + basename) if n >= len(img_list[:rang[0]+rang[1]]) else None

	for n, i in tqdm(enumerate(txt_list)):
		basename = os.path.basename(i)
		shutil.move(i, './labels/train/' + basename) if n < len(img_list[:rang[0]]) else None
		shutil.move(i, './labels/val/' + basename) if n >= len(img_list[:rang[0]]) and n < len(img_list[:rang[0]+rang[1]]) else None
		shutil.move(i, './labels/test/' + basename) if n >= len(img_list[:rang[0]+rang[1]]) else None

#------------------------------------------------------------------------------------
### Write txt file
def write_txt():
	print('[*] Start Write TXT')
	train_list = glob('./images/train/**.jpg')
	with open('train.txt', 'w') as f:
		for i in train_list:
			f.write(i+'\n')

	test_list = glob('./images/test/**.jpg')
	with open('test.txt', 'w') as f:
		for i in test_list:
			f.write(i+'\n')

	val_list = glob('./images/val/**.jpg')
	with open('val.txt', 'w') as f:
		for i in val_list:
			f.write(i+'\n')


if __name__ == '__main__':
	seperate_file()
	write_txt()