import glob

if __name__ == '__main__':
    dataset_path = "D:\\Dataset_TFE\\Images"
    count = 0

    # Anomalies spotted
    anomalies = 5

    # Check for duplicates
    check = []
    duplicates = []

    # Check for images with two versions
    extra = []

    for sub_dir in glob.glob(dataset_path + "/*/"):
        for filepath in glob.glob(sub_dir + '*.tif'):
        	name = filepath.split("\\")[-1]

        	if name[-6:-4] == '-1' or name[-6:-4] == '-2' or name[-6:-4] == '_1' or name[-6:-4] == '_2':
        		extra.append(name)

        	if name not in check:
        		check.append(name)
        	else:
        		duplicates.append(name)

        	count += 1

    print(f'Number of duplicates = {len(duplicates)}')
    print(f'Number of expected duplicates = {(641-569)*2 + 10}')


    nb_extra = int(len(extra)/2)
    print(f'Number of extra images = {nb_extra}')

    print(f'Sum of all images = {count}')
    print(f'Expected amount = {(945 + (641-569)) * 2 + nb_extra + anomalies}')

    # Print which number has extras
    numbers = []
    for ex in extra:
    	if ex[5:-8] not in numbers:
    		numbers.append(ex[5:-8])

    print(numbers)
    print(len(numbers))


    print()    
    # Get ids that are not in the annotaitons
    ids_d = []
    ids_v = []
    for name in check:
        if name[-6:-4] == '-1' or name[-6:-4] == '-2' or name[-6:-4] == '_1' or name[-6:-4] == '_2':
            if name[-8:-6] == '-v':
                ids_v.append(name[5:9])
            elif name[-8:-6] == '-d':
                ids_d.append(name[5:9])
        else:
            if name[-6:-4] == '-v':
                ids_v.append(name[5:9])
            elif name[-6:-4] == '-d':
                ids_d.append(name[5:9])
                

    ref = [str(nb).zfill(4) for nb in range(1, 946)]
    no_annot_d = list(set(ref) - set(ids_d))
    print(no_annot_d)
    print()
    no_annot_v = list(set(ref) - set(ids_v))
    print(no_annot_v)
