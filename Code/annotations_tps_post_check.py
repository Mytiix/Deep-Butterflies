def get_dict_annot_post(annot_path, filename):
	lines = []
	with open(annot_path + filename) as f:
		lines = f.readlines()


	# {img_name : [[[landmarks1], [landmarks2]], lm, scale]}
	coordinates = {}

	# [[landmarks1], [landmarks2]]
	landmarks = [[], []]
	pos = 0

	tab = False

	lm = 0
	scale = 0.0
	
	for line in lines:
		tmp = line.split()

		#print(tmp)

		# Coordinate
		if len(tmp) == 2:
			landmarks[pos].append(tuple(tmp))
			tab = False
		
		# Info
		elif len(tmp) == 1:
			if tmp[0][:3] == 'LM=':
				lm = tmp[0][3:]
			elif tmp[0][:6] == 'IMAGE=':
				img_name = tmp[0][6:]
				continue
			elif tmp[0][:6] == 'SCALE=':
				scale = tmp[0][6:]

			if pos == 1:
				# Add image coord
				coordinates[img_name] = [landmarks, lm, scale]

				# Reset landmarks
				landmarks = [[], [], []]
				pos = 0

			tab = False

		# Tabs
		elif len(tmp) == 0:
			if not tab:
				pos += 1
			tab = True

		# Anomaly
		else:
			print(tmp)

	return coordinates


if __name__ == '__main__':
	annot_path = "D:\\Dataset_TFE\\Annot\\"
	filename = "Concat_Aile_Post_Gauche[260].TPS"

	coordinates = get_dict_annot_post(annot_path, filename)	

	ids = [name[5:9] for name in coordinates.keys()]
	ref = [str(nb).zfill(4) for nb in range(1, 946)]

	no_annot = list(set(ref) - set(ids))

	print()
	print()

	print(ids)
	print(len(ids))
	print(len(set(ids)))

	print()
	print()

	print(len(no_annot))
	print(no_annot)

	print()

	# Check for extra
	extra = []

	for key in coordinates.keys():
		if key[-6:-4] == '-1' or key[-6:-4] == '-2' or key[-6:-4] == '_1' or key[-6:-4] == '_2':
			extra.append(key[5:-8])

	print(len(extra))
	print(extra)

	print()

	# Check content
	for value in coordinates.items():

		# Check number of landmarks
		if len(value[1][0][0]) != 19 or len(value[1][0][1]) != 10:
			print(value[0])

		if value[1][1] != '29':
			print(value[0])

	'''
	l=[]
	for key in coordinates.keys():
		l.append(key[5:9])

	print(l)
	'''