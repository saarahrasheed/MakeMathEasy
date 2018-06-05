def check_if_subset(setA, setB):
	# ensure the lists are identical
	setA = list(set(setA))
	setB = list(set(setB))
	countA = 0
	countB = 0
	for company in range(len(setA)):
		if setA[company] in setB:
			countA+=1
		else:
			continue
	if countA == len(setA):
		print("Set A is a subset of Set B")
	else:
		for company in range(len(setB)):
			if setB[company] in setA:
				countB+=1
			else:
				continue
		if countB == len(setB):
			print("Set B is a subset of A")
	return
