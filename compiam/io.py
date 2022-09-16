import csv

def write_csv(data, out_path, header=None):
	D = list(zip(*data))
	with open(out_path, 'w') as f:
		writer = csv.writer(f)
		if header:
			assert len(header) == len(D[0]), "Header and row length mismatch"
			writer.writerow(header)
		for row in D:
			writer.writerow(row)
