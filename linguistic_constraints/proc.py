pairs = []
with open("ppdb_synonyms.txt") as f:
	for line in f:
		pairs.append(" ".join(sorted(line.split(" "))))
		
pairs = list(set(pairs))
with open("synonyms.txt","w") as of:
	of.write("\n".join(pairs))
