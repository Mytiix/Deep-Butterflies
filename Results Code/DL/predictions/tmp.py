
if __name__ == '__main__':

	species = 'all'
	side = 'v'

	if side == 'd':
		filename = "ClassifierAA_2021[258].txt"
	else:
		filename = "ClassifierAP_2021[259].txt"

	lines = []
	with open(filename) as f:
		l = f.readlines()
		header = l[0]
		lines = l[1:]

	name_to_id = {}
	for i, line in enumerate(lines):
		split = line.split()
		name_to_id[split[1]] = i

	tps_filename = species+'_'+side+'.TPS'
	txt_filename = species+'_'+side+'_categories.txt'
	file = open(txt_filename, 'w')
	file.write(header)
	with open(tps_filename) as f:
		for line in f.readlines():
			if line[:6] == 'IMAGE=':
				name = line[6:10] + '-' + line[11:15]
				try:
					file.write(lines[name_to_id[name]])
				except KeyError:
					file.write(f'?\t{name}\t?\tMorpho\t?\t\t\t\t\n')


	file.close()