from collections import defaultdict
import re
alignment_1=[]
with open("upload_iter20_0.0005_Null_3_reverseST.a") as f:
	for line in f:
		line = line.strip().split()
		alignment_1.append([i for i in line])

#print alignment_1

alignment_2=[]

with open("upload_iter20_0.0005_Null_3.a") as e:
	for line in e:
		line = line.strip().split()
		alignment_2.append([i for i in line])
#print alignment_2
new_alignment=[]
for x in range(len(alignment_2)):
	intersection= set(alignment_2[x]).intersection(set(alignment_1[x]))
	line = []
	for e in intersection:
		line.append(e)
	new_alignment.append(line)
for aligment in new_alignment:
	a= defaultdict(lambda: "NULL")

	for align in aligment:
		m=align.split('-')[0]
		a[int(m)]=align
	for key in sorted(a):
		print a[key],
	print

	
