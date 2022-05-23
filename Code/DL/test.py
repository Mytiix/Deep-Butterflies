import numpy as np
import glob


def resort(file):
	lm_ids = [539978496, 539978479, 539978462, 539978445, 539978430, 539978413, 539978392, 539978369, 539978308,
			 539978343, 539978293, 539978547, 539978278, 539978532, 539978259, 539978222, 539978517, 539978191]
	s = {x : i for i, x in enumerate(lm_ids)}
	
	content = np.loadtxt(file)
		
	new_content = sorted(content, key=lambda x : s[x[0]])
	str_content = []
	for c in new_content:
		str_list = []
		for i, x in enumerate(c):
			if i < 3:
				str_list.append(str(int(x)))
			else:
				str_list.append(str(x))
		str_content.append(" ".join(str_list))

	new_file = open(file, 'w')
	new_file.write("\n".join(str_content))
	new_file.close()


if __name__ == '__main__':
	repository = 'D:/Dataset_TFE/images_v2/'

	lm_ids = [539978496, 539978479, 539978462, 539978445, 539978430, 539978413, 539978392, 539978369, 539978308,
			 539978343, 539978293, 539978547, 539978278, 539978532, 539978259, 539978222, 539978517, 539978191]
	# # lm_ids = [539978496, 539978479, 539978462, 539978445, 539978430, 539978413, 539978392, 539978369, 539978308,
	# # 		 539978343, 539978293, 539978278, 539978547, 539978259, 539978532, 539978222, 539978517, 539978191]

	# # lm_ids = [540037091, 540037085, 540037077, 540037069, 540037059, 540037053, 540037045, 540037039, 540037031,
	# # 		540037005, 540036993, 540036985, 540036979, 540036969]

	subspecies = 'polyphemus'

	for file in glob.glob(repository+'/'+subspecies+'/d/testing/landmarks_v2/*.txt'):
		resort(file)
	for file in glob.glob(repository+'/'+subspecies+'/d/training/landmarks_v2/*.txt'):
		resort(file)
	for file in glob.glob(repository+'/'+subspecies+'/d/landmarks_v2/*.txt'):
		resort(file)

	for folder in glob.glob(repository+'/'+subspecies+'/d/landmarks_v2/*.txt'):
		file = np.loadtxt(folder)
		for i, f in enumerate(file):
			if lm_ids[i] != int(f[0]):
				print(folder, i+1)



	# test = np.loadtxt('535765296.txt')
	# test = test[:,1:3]

	# print(test)
