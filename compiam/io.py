import csv

def write_csv(data, out_path, header=None):
	with open(out_file, 'w') as f:
		writer = csv.writer(f)
		if header:
			assert len(header) == len(row), "Header and row length mismatch"
			writer.writerow(header)
		for row in zip(*data):
			writer.writerow(row)
