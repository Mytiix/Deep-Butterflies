
if __name__ == '__main__':
	annot_path = "D:\\Dataset_TFE\\Annot\\"

	filename2 = "ClassifierAA_2021[258].txt"
	filename1 = "ClassifierAP_2021[259].txt"

	lines = []
	with open(annot_path + filename1) as f1:
		lines = f1.readlines()[1:]

	with open(annot_path + filename2) as f2:
		lines += f2.readlines()[1:]


	# Get possible values and count them
	check = {}
	anomalies = []

	sp = {}
	genre = {}
	sexe = {}

	for line in lines:
		# [ID, Individu, sexe, genre, sp]
		split = line.split()

		if split[1] not in check.keys():
			check[split[1]] = split[1:]
		else:
			if check[split[1]] != split[1:]:
				anomalies.append(split[1:])
			continue

		if split[4] not in sp.keys():
			sp[split[4]] = 1
		else:
			sp[split[4]] += 1

		if split[3] not in genre.keys():
			genre[split[3]] = 1
		else:
			genre[split[3]] += 1

		if split[2] not in sexe.keys():
			sexe[split[2]] = 1
		else:
			sexe[split[2]] += 1

	print(sexe)
	print(genre)
	print(sp)


	# Get ids that are not in the annotaitons
	ids = [name[-4:] for name in check.keys()]
	ref = [str(nb).zfill(4) for nb in range(1, 946)]
	no_annot = list(set(ref) - set(ids))
	no_annot.sort()


	print(no_annot)
	print(len(no_annot))
	print(945-len(no_annot))

	'''
	for anom in anomalies:
		print(anom)
		print(check[anom[0]])
		print()
	'''